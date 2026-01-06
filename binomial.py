import matplotlib.pyplot as plt
import numpy as np


def simple_binomial_tree(S, K, T, r, up, down, option_type='call'):
	u = up
	d = down
	a = np.exp(r * T)
	p = (S * a - d) / (u - d)

	# 1. 모든 노드의 주가와 옵션 가격 계산 (2차원 배열 구조)
	stock_tree = np.zeros((2, 2))
	option_tree = np.zeros((2, 2))

	stock_tree[0, 0] = S
	stock_tree[0, 1] = up
	stock_tree[1, 1] = down

	print(f"stock tree: {stock_tree}")

	# 만기 옵션 가치
	if option_type == 'call':
		option_tree[:, 1] = np.maximum(stock_tree[:, 1] - K, 0)
	else:
		option_tree[:, 1] = np.maximum(K - stock_tree[:, 1], 0)

	option_tree[0, 0] = np.exp(-r * T) * (p * option_tree[0, 1] + (1 - p) * option_tree[1, 1])

	print(f"option tree:{option_tree}")

	# --- 3. 시각화 ---
	fig, ax = plt.subplots(figsize=(13, 8))

	# 배경에 u, d, p 값 표시 (Legend 대용)
	param_text = (f"Parameters:\n"
				  f"u (Up): {u:.4f}\n"
				  f"d (Down): {d:.4f}\n"
				  f"p (Risk-neutral Prob): {p:.4f}\n"
				  f"T (Maturity): {T:.4f}")

	# 그래프 내부에 텍스트 박스 추가
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax.text(0.02, 0.95, param_text, transform=ax.transAxes, fontsize=10,
			verticalalignment='top', bbox=props)

	for i in range(2):
		for j in range(i + 1):  # i <= j 까지만 반복한다
			x, y = i, stock_tree[j, i]
			box_color = 'white'
			text_color = 'black'

			label = f"S:{y:.2f}\nV:{option_tree[j, i]:.2f}"
			ax.text(x + 0.1, y, label, verticalalignment='center', fontsize=9, fontweight='bold',
					bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'),
					color=text_color)

			ax.plot(x, y, 'ko', markersize=4, zorder=5)
			if i < 1:
				ax.plot([i, i + 1], [y, stock_tree[j, i + 1]], 'gray', alpha=0.2)
				ax.plot([i, i + 1], [y, stock_tree[j + 1, i + 1]], 'gray', alpha=0.2)

	plt.title(f"European {option_type.capitalize()} Option Tree (n=1)")
	plt.xlabel("Steps")
	plt.ylabel("Stock Price")
	plt.xlim(-0.5, 1 + 1.5)
	plt.grid(True, alpha=0.2)
	return option_tree[0, 0]


def european_binomial_tree(S, K, T, r, n, q=0.0, sigma=None, u_pct=None, d_pct=None, option_type='call'):
	dt = T / n

	# --- 파라미터 결정 로직 ---
	if u_pct is not None and d_pct is not None:
		# 사용자가 직접 퍼센트를 입력한 경우 (예: 0.1, -0.1)
		u = 1 + u_pct
		d = 1 + d_pct
		mode_desc = f"Input: (u:+{u_pct * 100}%, d:{d_pct * 100}%)"
	elif sigma is not None:
		# 변동성(sigma)을 기반으로 계산하는 경우 (CRR 모델)
		u = np.exp(sigma * np.sqrt(dt))
		d = 1 / u
		mode_desc = f"Input: volatility ({sigma * 100}%)"
	else:
		raise ValueError("변동성또는 퍼센트지 중 하나는 반드시 입력해야 합니다.")

	# 위험중립확률 p 계산
	p = (np.exp((r - q) * dt) - d) / (u - d)

	# --- 트리 생성 및 옵션가 계산 ---
	stock_tree = np.zeros((n + 1, n + 1))
	option_tree = np.zeros((n + 1, n + 1))

	for i in range(n + 1):
		for j in range(i + 1):
			stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

	if option_type == 'call':
		option_tree[:, n] = np.maximum(stock_tree[:, n] - K, 0)
	else:
		option_tree[:, n] = np.maximum(K - stock_tree[:, n], 0)

	for i in range(n - 1, -1, -1):
		for j in range(i + 1):
			option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])

	# --- 시각화 ---
	if n >= 15:
		print(f"크기(n={n})가 너무 큽니다. 시각화하지 않고 결과값 {option_tree[0, 0]:.3f}만 리턴합니다.")
		return option_tree[0, 0]

	fig, ax = plt.subplots(figsize=(12, 7))
	param_text = (f"{mode_desc}\n"
				  f"u: {u:.4f}, d: {d:.4f}\n"
				  f"p: {p:.4f}, r: {r * 100}%")

	ax.text(0.02, 0.95, param_text, transform=ax.transAxes, fontsize=10,
			verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

	for i in range(n + 1):
		for j in range(i + 1):
			x, y = i, stock_tree[j, i]
			box_color = 'white'
			text_color = 'black'

			ax.text(x + 0.1, y, f"S:{y:.1f}\nV:{option_tree[j, i]:.2f}",
					verticalalignment='center', fontsize=8, fontweight='bold',
					bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='gray', boxstyle='round'),
					color=text_color)

			ax.plot(x, y, 'ko', markersize=3, zorder=5)
			if i < n:
				ax.plot([i, i + 1], [y, stock_tree[j, i + 1]], 'gray', alpha=0.15)
				ax.plot([i, i + 1], [y, stock_tree[j + 1, i + 1]], 'gray', alpha=0.15)

	plt.title(f"European {option_type.capitalize()} Tree")
	plt.xlabel("Steps")
	plt.ylabel("Stock Price")
	plt.xlim(-0.5, n + 0.5)
	plt.grid(True, alpha=0.2)
	return option_tree[0, 0]


def american_binomial_tree(S, K, T, r, n, q=0.0, sigma=None, u_pct=None, d_pct=None, option_type='call'):
	# --- 1. 파라미터 계산 ---
	dt = T / n
	if u_pct is not None and d_pct is not None:
		# 사용자가 직접 퍼센트를 입력한 경우 (예: 0.1, -0.1)
		u = 1 + u_pct
		d = 1 + d_pct
		mode_desc = f"Input: (u:+{u_pct * 100}%, d:{d_pct * 100}%)"
	elif sigma is not None:
		# 변동성(sigma)을 기반으로 계산하는 경우 (CRR 모델)
		u = np.exp(sigma * np.sqrt(dt))
		d = 1 / u
		mode_desc = f"Input: volatility ({sigma * 100}%)"
	else:
		raise ValueError("변동성 또는 퍼센트 중 하나는 반드시 입력해야 합니다.")

	# 위험중립확률 p 계산
	p = (np.exp((r - q) * dt) - d) / (u - d)

	# --- 2. 트리 데이터 생성 (이전 로직과 동일) ---
	stock_tree = np.zeros((n + 1, n + 1))
	option_tree = np.zeros((n + 1, n + 1))
	early_exercise = np.zeros((n + 1, n + 1), dtype=bool)

	for i in range(n + 1):
		for j in range(i + 1):
			stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

	if option_type == 'call':
		option_tree[:, n] = np.maximum(stock_tree[:, n] - K, 0)
	else:
		option_tree[:, n] = np.maximum(K - stock_tree[:, n], 0)

	for i in range(n - 1, -1, -1):
		for j in range(i + 1):
			continuation_val = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
			exercise_val = np.maximum(stock_tree[j, i] - K, 0) if option_type == 'call' else np.maximum(
				K - stock_tree[j, i], 0)

			if exercise_val > continuation_val:
				option_tree[j, i] = exercise_val
				early_exercise[j, i] = True
			else:
				option_tree[j, i] = continuation_val

	# --- 3. 시각화 ---
	if n >= 15:
		print(f"크기(n={n})가 너무 큽니다. 시각화하지 않고 결과값 {option_tree[0, 0]:.3f}만 리턴합니다.")
		return option_tree[0, 0]

	fig, ax = plt.subplots(figsize=(13, 8))

	# 배경에 u, d, p 값 표시 (Legend 대용)
	param_text = (f"Parameters:\n"
				  f"u (Up): {u:.4f}\n"
				  f"d (Down): {d:.4f}\n"
				  f"p (Risk-neutral Prob): {p:.4f}\n"
				  f"dt (Time step): {dt:.4f}")

	# 그래프 내부에 텍스트 박스 추가
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax.text(0.02, 0.95, param_text, transform=ax.transAxes, fontsize=10,
			verticalalignment='top', bbox=props)

	for i in range(n + 1):
		for j in range(i + 1):
			x, y = i, stock_tree[j, i]
			box_color = 'tomato' if early_exercise[j, i] else 'white'
			text_color = 'white' if early_exercise[j, i] else 'black'

			label = f"S:{y:.2f}\nV:{option_tree[j, i]:.2f}"
			ax.text(x + 0.1, y, label, verticalalignment='center', fontsize=9, fontweight='bold',
					bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'),
					color=text_color)

			ax.plot(x, y, 'ko', markersize=4, zorder=5)
			if i < n:
				ax.plot([i, i + 1], [y, stock_tree[j, i + 1]], 'gray', alpha=0.2)
				ax.plot([i, i + 1], [y, stock_tree[j + 1, i + 1]], 'gray', alpha=0.2)

	plt.title(f"American {option_type.capitalize()} Option Tree (n={n})")
	plt.xlabel("Steps")
	plt.ylabel("Stock Price")
	plt.xlim(-0.5, n + 1.5)
	plt.grid(True, alpha=0.2)
	plt.show()
	return option_tree[0, 0]

if __name__ == "__main__":
	# --- 테스트 ---
	# 1. 상승 10%, 하락 10% 직접 입력 방식
	european_binomial_tree(S=100, K=105, T=1, r=0.05, n=4, u_pct=0.1, d_pct=-0.1, option_type='put')
	# 실행
	american_binomial_tree(S=100, K=110, T=0.5, r=0.1, sigma=0.3, n=4, option_type="put")