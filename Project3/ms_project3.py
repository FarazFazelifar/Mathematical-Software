import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

# ------- بخش ۱: درون‌یابی لاگرانژ و اسپلاین -------
print("=" * 50)
print("بخش ۱: درون‌یابی لاگرانژ و اسپلاین مکعبی")
print("=" * 50)

# نقاط داده‌شده
points = [(1, 1), (2, 3), (3, 5), (4, 8), (5, 5), (6, 2)]
xs = np.array([p[0] for p in points])
ys = np.array([p[1] for p in points])
n = len(points)

# تابع پایه لاگرانژ
def lagrange_basis(i, x, xn):
    L = 1.0
    for j in range(len(xn)):
        if j != i:
            L *= (x - xn[j]) / (xn[i] - xn[j])
    return L

# محاسبه چندجمله‌ای لاگرانژ
def lagrange_interp(x, xn, yn):
    s = 0.0
    for i in range(len(xn)):
        s += yn[i] * lagrange_basis(i, x, xn)
    return s

# ارزیابی روی بازه ریز
x_fine = np.linspace(1, 6, 200)
y_lag = np.array([lagrange_interp(x, xs, ys) for x in x_fine])

# ضرایب چندجمله‌ای
coeffs = np.polyfit(xs, ys, n - 1)
print(f"ضرایب لاگرانژ (بالاترین درجه تا پایین): {np.round(coeffs, 4)}")

# ساخت رشته ضابطه
poly_str = "P(x) = "
for i, c in enumerate(coeffs):
    deg = n - 1 - i
    if abs(c) < 1e-10:
        continue
    sign = "+" if c >= 0 and i > 0 else ""
    if deg == 0:
        poly_str += f"{sign}{c:.4f}"
    elif deg == 1:
        poly_str += f"{sign}{c:.4f}x"
    else:
        poly_str += f"{sign}{c:.4f}x^{deg}"

print(f"ضابطه: {poly_str}")

# بررسی خطا در نقاط
err = max(abs(lagrange_interp(xs[i], xs, ys) - ys[i]) for i in range(n))
print(f"بیشترین خطا در نقاط گره: {err:.2e}")

# اسپلاین مکعبی طبیعی
cs = CubicSpline(xs, ys, bc_type='natural')
y_spl = cs(x_fine)

print("\nضرایب اسپلاین (هر بازه):")
for i in range(n - 1):
    c = cs.c[:, i]
    print(f"  [{xs[i]:.0f}, {xs[i+1]:.0f}]: d={c[0]:.4f}, c={c[1]:.4f}, b={c[2]:.4f}, a={c[3]:.4f}")

# رسم نمودار مقایسه‌ای
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(xs, ys, color='red', s=80, zorder=5, label='Data Points')
ax.plot(x_fine, y_lag, 'b-', lw=2, label='Lagrange (deg 5)')
ax.plot(x_fine, y_spl, 'g--', lw=2, label='Natural Cubic Spline')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Lagrange vs Cubic Spline Interpolation')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig('interpolation_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n-> interpolation_plot.png saved")


# ------- بخش ۲: تحلیل داده اکسل -------
print("\n" + "=" * 50)
print("بخش ۲: تحلیل فایل اکسل")
print("=" * 50)

# خواندن فایل اکسل
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(script_dir, 'پروژه 3.xlsx')
df = pd.read_excel(excel_file, engine='openpyxl')

# نام ستون‌ها
size_col = df.columns[0]
a1 = df.columns[1]
a2 = df.columns[2]
a3 = df.columns[3]

print(f"\nداده‌ها ({len(df)} سطر):")
print(df.to_string(index=False))

# الف) نمودارهای میله‌ای، خطی و جعبه‌ای

# نمودار میله‌ای
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(df))
w = 0.25
ax.bar(x_pos - w, df[a1], w, label='Alg.1', color='#4CAF50')
ax.bar(x_pos, df[a2], w, label='Alg.2', color='#2196F3')
ax.bar(x_pos + w, df[a3], w, label='Alg.3', color='#FF9800')
ax.set_xlabel('Data Size')
ax.set_ylabel('Runtime (ms)')
ax.set_title('Bar Chart: Algorithm Runtimes')
ax.set_xticks(x_pos)
ax.set_xticklabels(df[size_col], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('bar_plot.png', dpi=150)
plt.close()
print("\n-> bar_plot.png saved")

# نمودار خطی
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df[size_col], df[a1], 'o-', color='#4CAF50', lw=2, ms=8, label='Alg.1')
ax.plot(df[size_col], df[a2], 's-', color='#2196F3', lw=2, ms=8, label='Alg.2')
ax.plot(df[size_col], df[a3], '^-', color='#FF9800', lw=2, ms=8, label='Alg.3')
ax.set_xlabel('Data Size')
ax.set_ylabel('Runtime (ms)')
ax.set_title('Line Chart: Runtime Trends')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df[size_col], rotation=45)
fig.tight_layout()
fig.savefig('line_plot.png', dpi=150)
plt.close()
print("-> line_plot.png saved")

# نمودار جعبه‌ای
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#4CAF50', '#2196F3', '#FF9800']
bp = ax.boxplot([df[a1].values, df[a2].values, df[a3].values],
                tick_labels=['Alg.1', 'Alg.2', 'Alg.3'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Box Plot: Runtime Distributions')
ax.set_ylabel('Runtime (ms)')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('box_plot.png', dpi=150)
plt.close()
print("-> box_plot.png saved")

# ب) افزودن داده جدید و آپدیت نمودارها
print("\\n" + "=" * 50)
print("بخش ۲ب: افزودن داده جدید (700KB) و آپدیت خودکار")
print("=" * 50)

# ایجاد یک کپی با سطر جدید
df2 = pd.concat([df, pd.DataFrame({
    size_col: ['700KB'],
    a1: [80],
    a2: [320],
    a3: [700],
})], ignore_index=True)

print(f"\\nداده‌های آپدیت‌شده ({len(df2)} سطر):")
print(df2.to_string(index=False))

# آپدیت میله‌ای
fig, ax = plt.subplots(figsize=(10, 6))
x_pos2 = np.arange(len(df2))
ax.bar(x_pos2 - w, df2[a1], w, label='Alg.1', color='#4CAF50')
ax.bar(x_pos2, df2[a2], w, label='Alg.2', color='#2196F3')
ax.bar(x_pos2 + w, df2[a3], w, label='Alg.3', color='#FF9800')
ax.set_xlabel('Data Size')
ax.set_ylabel('Runtime (ms)')
ax.set_title('Bar Chart (Updated with 700KB)')
ax.set_xticks(x_pos2)
ax.set_xticklabels(df2[size_col], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('bar_plot_updated.png', dpi=150)
plt.close()
print("-> bar_plot_updated.png saved")

# آپدیت خطی
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df2[size_col], df2[a1], 'o-', color='#4CAF50', lw=2, ms=8, label='Alg.1')
ax.plot(df2[size_col], df2[a2], 's-', color='#2196F3', lw=2, ms=8, label='Alg.2')
ax.plot(df2[size_col], df2[a3], '^-', color='#FF9800', lw=2, ms=8, label='Alg.3')
ax.set_xlabel('Data Size')
ax.set_ylabel('Runtime (ms)')
ax.set_title('Line Chart (Updated with 700KB)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(len(df2)))
ax.set_xticklabels(df2[size_col], rotation=45)
fig.tight_layout()
fig.savefig('line_plot_updated.png', dpi=150)
plt.close()
print("-> line_plot_updated.png saved")

# ج) میانگین الگوریتم ۲
alg2_vals = df[a2].values
alg2_mean = np.mean(alg2_vals)
print(f"\nمقادیر Alg.2: {alg2_vals}")
print(f"میانگین Alg.2: {alg2_mean:.2f} ms")


# ------- بخش ۳: انتگرال‌گیری عددی -------
print("\n" + "=" * 50)
print("بخش ۳: انتگرال‌گیری عددی — f(x) = e^x روی [0, 1]")
print("=" * 50)

def f(x):
    return np.exp(x)

exact = np.e - 1  # جواب دقیق
print(f"مقدار دقیق: e - 1 = {exact:.15f}")

# الف) روش سیمپسون ترکیبی
def simpson(f, a, b, n):
    # n باید زوج باشد
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = f(x)
    return (h / 3) * (fx[0] + fx[-1] + 4*np.sum(fx[1:-1:2]) + 2*np.sum(fx[2:-2:2]))

# روش ذوزنقه‌ای ترکیبی
def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = f(x)
    return h * (0.5*fx[0] + 0.5*fx[-1] + np.sum(fx[1:-1]))

print("\n--- سیمپسون ---")
for nn in [4, 8, 16, 32, 64, 128]:
    val = simpson(f, 0, 1, nn)
    print(f"  n={nn:3d}: {val:.12f}  خطا = {abs(val - exact):.2e}")

print("\n--- ذوزنقه‌ای ---")
for nn in [4, 8, 16, 32, 64, 128]:
    val = trapezoidal(f, 0, 1, nn)
    print(f"  n={nn:3d}: {val:.12f}  خطا = {abs(val - exact):.2e}")

# ب) انتگرال گوسی-لژاندر
from numpy.polynomial.legendre import leggauss

def gauss_legendre(f, a, b, n):
    xi, w = leggauss(n)
    # نگاشت از [-1,1] به [a,b]
    x_mapped = 0.5*(b - a)*xi + 0.5*(b + a)
    return 0.5*(b - a) * np.sum(w * f(x_mapped))

print("\n--- گوسی-لژاندر ---")
for nn in [1, 2, 3, 4, 5, 6]:
    val = gauss_legendre(f, 0, 1, nn)
    print(f"  n={nn}: {val:.12f}  خطا = {abs(val - exact):.2e}")

# جدول مقایسه
s16 = simpson(f, 0, 1, 16)
t16 = trapezoidal(f, 0, 1, 16)
g5 = gauss_legendre(f, 0, 1, 5)

print(f"\n{'='*50}")
print("مقایسه (n=16 برای سیمپسون/ذوزنقه، n=5 برای گوسی)")
print(f"{'='*50}")
print(f"  سیمپسون (n=16)   : {s16:.15f}  خطا = {abs(s16-exact):.2e}")
print(f"  ذوزنقه‌ای (n=16)  : {t16:.15f}  خطا = {abs(t16-exact):.2e}")
print(f"  گوسی-لژاندر (n=5): {g5:.15f}  خطا = {abs(g5-exact):.2e}")
print(f"  دقیق              : {exact:.15f}")

# نمودار همگرایی
fig, ax = plt.subplots(figsize=(10, 6))
ns = np.arange(2, 65, 2)
errs_s = [abs(simpson(f, 0, 1, n) - exact) for n in ns]
errs_t = [abs(trapezoidal(f, 0, 1, n) - exact) for n in ns]
ax.semilogy(ns, errs_s, 'b-o', ms=4, label='Simpson')
ax.semilogy(ns, errs_t, 'r-s', ms=4, label='Trapezoidal')
ax.set_xlabel('n (subintervals)')
ax.set_ylabel('Absolute Error')
ax.set_title('Convergence Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('integration_convergence.png', dpi=150)
plt.close()
print("\n-> integration_convergence.png saved")

print("\n" + "=" * 50)
print("تمام بخش‌ها انجام شد.")
print("=" * 50)
