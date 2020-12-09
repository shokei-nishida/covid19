# covid19
コロナウイルス感染者シミュレーションだじょ～ブフオォｗｗｗｗｗｗｗｗｗ

import numpy as np                  # 配列を扱う数値計算ライブラリNumPy
import matplotlib.pyplot as plt     # グラフ描画ライブラリmatplotlib
import japanize_matplotlib          # matplotlibの日本語化

# （2）時間変数tの導入
T = 700                 # 変数tの範囲 0≦t<T（日）（250,150,150,700と値を変えてシミュレーションを行う）
n = 10*T                 # 変数tの範囲をn等分   n=T/h=T/0.1=10*T （T=250のときはn=2500）
h = 0.1                  # 等差数列の公差:0.1 固定
t = np.arange(0,T,h)     # 0から公差dtでTを超えない範囲で等差数列を生成 t[0],...,t[n-1] 要素数n個

# （3）SIRモデル
# 3-1パラメータ
N = 7000000000         # モデルエリアの人口（人）（東京都1400万人に匹敵するエリアを想定） N=S+I+R＝一定
m = 50                 # 1日1人あたり接触する人数（人）（10,50,100,5と値を変えてシミュレーションを行う）
p = 0.02                 # 接触ごとに感染が生じる1日あたりの確率
d = 14                   # 感染者の回復平均日数（日）
beta = m*p / N           # 接触あたりの感染率
gamma = 1/d              # 回復率（隔離率）
# 3-2初期値
I_0 = 100                # 初期感染者数（人）100人
R_0 = 0                  # 初期回復者数（人）0人
S_0 = N - I_0 - R_0      # 初期未感染者数（人）S_0 = N - I_0 - R_0
# 3-3微分方程式
dSdt = lambda S, I, R, t : - beta*S*I               # dSdt ≡ dS/dt  dSdt(S, I, R, t)
dIdt = lambda S, I, R, t : beta*S*I - gamma*I       # dIdt ≡ dI/dt  dIdt(S, I, R, t)
dRdt = lambda S, I, R, t : gamma*I                  # dRdt ≡ dR/dt  dRdt(S ,I ,R, t)
# 3-4数値積分変数S,I,Rをリストとして生成
S = np.empty(n)          # S[0],...,S[n-1] 要素数n個
I = np.empty(n)          # I[0],...,I[n-1] 要素数n個
R = np.empty(n)          # R[0],...,R[n-1] 要素数n個
# 3-5初期値代入
S[0] = S_0
I[0] = I_0
R[0] = R_0

# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods 
#と、思ったが、ルンゲクッタわかんないのでオイラー法にしたったｗｗｗｗｗｗブフオォｗｗｗｗｗｗｗｗｗ

for j in range(n-1):     # j=0,...,n-2 -> S[j]=S[0],...,S[n-1]（I[j],R[j]も同じ） 要素数n個

    S[j+1] = S[j] +   h * dSdt( S[j] ,I[j] ,R[j] ,t[j] )# 末項 j=n-2 -> S[j+1]=S[n-1]
    I[j+1] = I[j] +  h * dIdt( S[j] ,I[j] ,R[j] ,t[j] )   # 末項 j=n-2 -> I[j+1]=I[n-1]
    R[j+1] = R[j] + h * dRdt( S[j] ,I[j] ,R[j] ,t[j] )
    
print(max(I))    
    
#     


# （5）結果表示 データプロットによるグラフ表示
# 点(t,S),点(t,I),点(t,R) それぞれ要素数n個のプロット
plt.plot(t, S, color = "green", label = "S:未感染者", linewidth = 1.0)
plt.plot(t, I, color = "red", label = "I:感染者", linewidth = 1.0)
plt.plot(t, R, color= "blue", label = "R:免疫獲得者", linewidth = 1.0)
# グラフの見た目設定
plt.title('SIRモデル RK4によるシミュレーション（m={},T={}）'.format(m,T))  # グラフタイトル パラメータmとTの値表示
plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
plt.gca().set_yticklabels(['{:.0f}%'.format(y/(N/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数（総人口1000万人に対する割合）')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()

