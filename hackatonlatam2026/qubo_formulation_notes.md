# QUBO Formulation Notes for Falcon Reservoir Release Scheduling

These notes summarize the QUBO formulation we developed for the Falcon Reservoir challenge.

---

## 1. Problem setup

The optimization variable is the release adjustment:

$$
u(t)$$

The optimized release is:

$$
R(t)=R^{obs}(t)+u(t)
$$

The simplified storage dynamics are:

$$
S_{opt}(t+1)=S_{opt}(t)+\Delta S^{obs}(t)-u(t)
$$

The Storage Resilience Score is based on:

$$
SRS=-(w_1C_{crit}+w_2C_{dev}+w_3C_{smooth})
$$

where:

$$
C_{crit}=\sum_{t=0}^{T}\left[\max(0,S_{min}-S_{opt}(t))\right]^2
$$

$$
C_{dev}=\sum_{t=0}^{T-1}u(t)^2
$$

$$
C_{smooth}=\sum_{t=1}^{T-1}[u(t)-u(t-1)]^2
$$

---

## 2. Binary encoding of release adjustment

The official benchmark uses five release-adjustment levels:

$$
u(t)\in\{-2\Delta u,-\Delta u,0,\Delta u,2\Delta u\}
$$

Use one-hot binary variables:

$$
x_{t,-2},x_{t,-1},x_{t,0},x_{t,1},x_{t,2}\in\{0,1\}
$$

with the one-hot condition:

$$
x_{t,-2}+x_{t,-1}+x_{t,0}+x_{t,1}+x_{t,2}=1
$$

Then:

$$
u(t)=\Delta u(-2x_{t,-2}-x_{t,-1}+0x_{t,0}+x_{t,1}+2x_{t,2})
$$

Define:

$$
a=[-2,-1,0,1,2]^T
$$

and:

$$
u(t)=\Delta u\,a^Tx_t
$$

The full binary vector is:

$$
x=
[x_{0,-2},x_{0,-1},x_{0,0},x_{0,1},x_{0,2},\ldots,x_{T-1,2}]^T
$$

This vector has length:

$$
5T
$$

The QUBO objective has the form:

$$
\min_x x^TQx
$$

---

## 3. One-hot constraint

For each time step:

$$
x_{t,-2}+x_{t,-1}+x_{t,0}+x_{t,1}+x_{t,2}=1
$$

Add the penalty:

$$
P_{onehot}=\lambda_{onehot}\sum_{t=0}^{T-1}
\left(x_{t,-2}+x_{t,-1}+x_{t,0}+x_{t,1}+x_{t,2}-1\right)^2
$$

For one week, using the variable order:

$$
[x_{t,-2},x_{t,-1},x_{t,0},x_{t,1},x_{t,2}]
$$

one upper-triangular QUBO block is:

$$
Q^{onehot}_t=
\lambda_{onehot}
\begin{bmatrix}
-1 & 2 & 2 & 2 & 2\\
0 & -1 & 2 & 2 & 2\\
0 & 0 & -1 & 2 & 2\\
0 & 0 & 0 & -1 & 2\\
0 & 0 & 0 & 0 & -1
\end{bmatrix}
$$

---

## 4. Deviation cost: \(C_{dev}\)

The deviation cost is:

$$
C_{dev}=\sum_{t=0}^{T-1}u(t)^2
$$

Substitute the binary encoding:

$$
C_{dev}=\Delta u^2\sum_{t=0}^{T-1}
(-2x_{t,-2}-x_{t,-1}+x_{t,1}+2x_{t,2})^2
$$

For one time step:

$$
\begin{aligned}
u(t)^2
=\Delta u^2(&
4x_{t,-2}+x_{t,-1}+x_{t,1}+4x_{t,2}\\
&+4x_{t,-2}x_{t,-1}
-4x_{t,-2}x_{t,1}
-8x_{t,-2}x_{t,2}\\
&-2x_{t,-1}x_{t,1}
-4x_{t,-1}x_{t,2}
+4x_{t,1}x_{t,2}).
\end{aligned}
$$

Using the upper-triangular convention, the one-week QUBO block is:

$$
Q^{dev}_t=w_2\Delta u^2
\begin{bmatrix}
4 & 4 & 0 & -4 & -8\\
0 & 1 & 0 & -2 & -4\\
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 4\\
0 & 0 & 0 & 0 & 4
\end{bmatrix}
$$

Since:

$$
w_2=\frac{0.1}{Tu_{max}^2},\qquad u_{max}=2\Delta u
$$

we get:

$$
w_2\Delta u^2=\frac{0.025}{T}
$$

Therefore:

$$
Q^{dev}_t=
\frac{0.025}{T}
\begin{bmatrix}
4 & 4 & 0 & -4 & -8\\
0 & 1 & 0 & -2 & -4\\
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 4\\
0 & 0 & 0 & 0 & 4
\end{bmatrix}
$$

The full matrix \(Q^{dev}\) is block diagonal with one copy of this block per time step.

---

## 5. Smoothness cost: \(C_{smooth}\)

The smoothness cost is:

$$
C_{smooth}=\sum_{t=1}^{T-1}[u(t)-u(t-1)]^2
$$

Using:

$$
u(t)=\Delta u\,a^Tx_t
$$

we get:

$$
[u(t)-u(t-1)]^2=
\Delta u^2(a^Tx_t-a^Tx_{t-1})^2
$$

Expanding:

$$
=\Delta u^2[(a^Tx_t)^2+(a^Tx_{t-1})^2-2(a^Tx_t)(a^Tx_{t-1})]
$$

Define:

$$
B=
\begin{bmatrix}
4 & 4 & 0 & -4 & -8\\
0 & 1 & 0 & -2 & -4\\
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 4\\
0 & 0 & 0 & 0 & 4
\end{bmatrix}
$$

and:

$$
C=
\begin{bmatrix}
-8 & -4 & 0 & 4 & 8\\
-4 & -2 & 0 & 2 & 4\\
0 & 0 & 0 & 0 & 0\\
4 & 2 & 0 & -2 & -4\\
8 & 4 & 0 & -4 & -8
\end{bmatrix}
$$

Then:

$$
Q^{smooth}=w_3\Delta u^2
\begin{bmatrix}
B & C & 0 & \cdots & 0\\
0 & 2B & C & \cdots & 0\\
0 & 0 & 2B & \cdots & 0\\
\vdots & \vdots & \vdots & \ddots & C\\
0 & 0 & 0 & 0 & B
\end{bmatrix}
$$

Using:

$$
w_3=\frac{0.1}{(T-1)(2u_{max})^2},\qquad u_{max}=2\Delta u
$$

we get:

$$
w_3\Delta u^2=\frac{0.1}{16(T-1)}
$$

Therefore:

$$
Q^{smooth}=\frac{0.1}{16(T-1)}
\begin{bmatrix}
B & C & 0 & \cdots & 0\\
0 & 2B & C & \cdots & 0\\
0 & 0 & 2B & \cdots & 0\\
\vdots & \vdots & \vdots & \ddots & C\\
0 & 0 & 0 & 0 & B
\end{bmatrix}
$$

---

## 6. Critical-storage cost: \(C_{crit}\)

The original term is:

$$
C_{crit}=\sum_{t=0}^{T}\left[\max(0,S_{min}-S_{opt}(t))\right]^2
$$

The storage equation is:

$$
S_{opt}(t)=S_0+\sum_{\tau=0}^{t-1}\Delta S^{obs}(\tau)-\sum_{\tau=0}^{t-1}u(\tau)
$$

Define the known constant:

$$
A_t=S_{min}-S_0-\sum_{\tau=0}^{t-1}\Delta S^{obs}(\tau)
$$

Then:

$$
S_{min}-S_{opt}(t)=A_t+\sum_{\tau=0}^{t-1}u(\tau)
$$

Ignoring the max temporarily, use:

$$
C_{crit}\approx\sum_{t=0}^{T}
\left(A_t+\sum_{\tau<t}u(\tau)\right)^2
$$

Substitute:

$$
u(\tau)=\Delta u\sum_k a_kx_{\tau,k}
$$

Then:

$$
C_{crit}\approx
\sum_{t=0}^{T}
\left(
A_t+\Delta u\sum_{\tau<t}\sum_k a_kx_{\tau,k}
\right)^2
$$

After expansion, constants can be ignored. The QUBO contributions are:

### Diagonal terms

$$
Q^{crit}_{(\tau,k),(\tau,k)}
+=
w_1\left[
2\Delta u\,a_k\sum_{t=\tau+1}^{T}A_t
+\Delta u^2a_k^2(T-\tau)
\right]
$$

### Off-diagonal terms

For \((\tau,k)\neq(\sigma,\ell)\):

$$
Q^{crit}_{(\tau,k),(\sigma,\ell)}
+=
2w_1\Delta u^2a_ka_\ell
\left(T-\max(\tau,\sigma)\right)
$$

The term \(T-\max(\tau,\sigma)\) counts how many future storage equations contain both release decisions.

---

## 7. Constraint \(R(t)\ge 0\)

Since:

$$
R(t)=R^{obs}(t)+u(t)
$$

we require:

$$
R^{obs}(t)+u(t)\ge 0
$$

The simple implementation is to penalize release choices that make this negative.

Define:

$$
I_{t,k}=\begin{cases}
1, & R^{obs}(t)+a_k\Delta u<0\\
0, & R^{obs}(t)+a_k\Delta u\ge 0
\end{cases}
$$

Then:

$$
P_R=\lambda_R\sum_{t=0}^{T-1}\sum_k I_{t,k}x_{t,k}
$$

So the QUBO contribution is diagonal only:

$$
Q^R_{(t,k),(t,k)}+=\lambda_RI_{t,k}
$$

---

## 8. Slack formulation for \(|u(t)|\le u_{max}\)

The constraint:

$$
|u(t)|\le u_{max}
$$

is equivalent to:

$$
u(t)+u_{max}\ge 0
$$

and:

$$
u_{max}-u(t)\ge 0
$$

Use slack variables:

$$
u(t)+u_{max}-s_t^+=0
$$

$$
u_{max}-u(t)-s_t^-=0
$$

The penalty is:

$$
P_u=\lambda_u\sum_{t=0}^{T-1}
\left(u(t)+u_{max}-s_t^+\right)^2
+
\lambda_u\sum_{t=0}^{T-1}
\left(u_{max}-u(t)-s_t^-\right)^2
$$

Encode:

$$
s_t^+=\Delta u(y_{t,0}^+ +2y_{t,1}^+ +4y_{t,2}^+)
$$

$$
s_t^-=\Delta u(y_{t,0}^- +2y_{t,1}^- +4y_{t,2}^-)
$$

For one time step, use variable order:

$$
[x_{t,-2},x_{t,-1},x_{t,0},x_{t,1},x_{t,2},y_{t,0}^+,y_{t,1}^+,y_{t,2}^+,y_{t,0}^-,y_{t,1}^-,y_{t,2}^-]
$$

Then:

$$
Q^{u\text{-slack}}_t=
\lambda_u\Delta u^2
\begin{bmatrix}
2B & C^+ & C^-\\
0 & D & 0\\
0 & 0 & D
\end{bmatrix}
$$

where:

$$
D=
\begin{bmatrix}
-3 & 4 & 8\\
0 & -4 & 16\\
0 & 0 & 0
\end{bmatrix}
$$

$$
C^+=
\begin{bmatrix}
4 & 8 & 16\\
2 & 4 & 8\\
0 & 0 & 0\\
-2 & -4 & -8\\
-4 & -8 & -16
\end{bmatrix}
$$

$$
C^-=
\begin{bmatrix}
-4 & -8 & -16\\
-2 & -4 & -8\\
0 & 0 & 0\\
2 & 4 & 8\\
4 & 8 & 16
\end{bmatrix}
$$

This constraint is redundant for the official encoding because \(u(t)\in[-2\Delta u,2\Delta u]\) and \(u_{max}=2\Delta u\), but it can still be included as a penalty.

---

## 9. Slack formulation for \(0\le S_{opt}(t)\le S_{max}\)

Split the constraint into:

$$
S_{opt}(t)\ge 0
$$

and:

$$
S_{opt}(t)\le S_{max}
$$

Use slack variables:

$$
S_{opt}(t)-s_t^{low}=0
$$

$$
S_{max}-S_{opt}(t)-s_t^{high}=0
$$

The penalty is:

$$
P_S=\lambda_S\sum_{t=0}^{T}
\left(S_{opt}(t)-s_t^{low}\right)^2
+
\lambda_S\sum_{t=0}^{T}
\left(S_{max}-S_{opt}(t)-s_t^{high}\right)^2
$$

Using:

$$
S_{opt}(t)=B_t-\sum_{\tau<t}u(\tau)
$$

where:

$$
B_t=S_0+\sum_{\tau=0}^{t-1}\Delta S^{obs}(\tau)
$$

we get:

$$
P_S=\lambda_S\sum_{t=0}^{T}
\left(B_t-\sum_{\tau<t}u(\tau)-s_t^{low}\right)^2
+
\lambda_S\sum_{t=0}^{T}
\left(S_{max}-B_t+\sum_{\tau<t}u(\tau)-s_t^{high}\right)^2
$$

Encode the storage slacks as:

$$
s_t^{low}=\Delta S\sum_{m=0}^{M_S-1}2^mz_{t,m}^{low}
$$

$$
s_t^{high}=\Delta S\sum_{m=0}^{M_S-1}2^mz_{t,m}^{high}
$$

Choose:

$$
M_S=\left\lceil\log_2\left(\frac{S_{max}}{\Delta S}+1\right)\right\rceil
$$

### General QUBO rule

For any expression:

$$
\left(c+\sum_i r_ib_i\right)^2
$$

add:

$$
Q_{i,i}+=\lambda(r_i^2+2cr_i)
$$

and for \(i<j\):

$$
Q_{i,j}+=2\lambda r_ir_j
$$

Apply this rule once to each low-storage constraint and once to each high-storage constraint.

---

## 10. Release-balance constraint

The original constraint is:

$$
\left|\sum_{t=0}^{T-1}u(t)\right|\le
\eta\sum_{t=0}^{T-1}R^{obs}(t)
$$

The simplified penalty we chose is:

$$
P_{bal}=\lambda_{bal}\left(\sum_{t=0}^{T-1}u(t)\right)^2
$$

Substitute:

$$
u(t)=\Delta u\sum_k a_kx_{t,k}
$$

Then:

$$
P_{bal}=\lambda_{bal}\Delta u^2
\left(\sum_{t=0}^{T-1}\sum_k a_kx_{t,k}\right)^2
$$

Therefore:

### Diagonal terms

$$
Q^{bal}_{(t,k),(t,k)}+=\lambda_{bal}\Delta u^2a_k^2
$$

### Off-diagonal terms

For \((t,k)<(\tau,\ell)\):

$$
Q^{bal}_{(t,k),(\tau,\ell)}+=2\lambda_{bal}\Delta u^2a_ka_\ell
$$

Equivalently:

$$
Q^{bal}=\lambda_{bal}\Delta u^2
\begin{bmatrix}
B & 2A & 2A & \cdots & 2A\\
0 & B & 2A & \cdots & 2A\\
0 & 0 & B & \cdots & 2A\\
\vdots & \vdots & \vdots & \ddots & 2A\\
0 & 0 & 0 & 0 & B
\end{bmatrix}
$$

where:

$$
A=aa^T=
\begin{bmatrix}
4 & 2 & 0 & -2 & -4\\
2 & 1 & 0 & -1 & -2\\
0 & 0 & 0 & 0 & 0\\
-2 & -1 & 0 & 1 & 2\\
-4 & -2 & 0 & 2 & 4
\end{bmatrix}
$$

and:

$$
B=
\begin{bmatrix}
4 & 4 & 0 & -4 & -8\\
0 & 1 & 0 & -2 & -4\\
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 4\\
0 & 0 & 0 & 0 & 4
\end{bmatrix}
$$

---

## 11. Final total QUBO

The final QUBO is:

$$
Q=
Q^{dev}
+Q^{smooth}
+Q^{crit}
+Q^{onehot}
+Q^R
+Q^{u\text{-slack}}
+Q^{S\text{-slack}}
+Q^{bal}
$$

and the optimization problem is:

$$
\min_x x^TQx
$$

After solving, decode each time step by selecting the active one-hot variable:

$$
u(t)=\Delta u(-2x_{t,-2}-x_{t,-1}+x_{t,1}+2x_{t,2})
$$

and then compute:

$$
R(t)=R^{obs}(t)+u(t)
$$

$$
S_{opt}(t+1)=S_{opt}(t)+\Delta S^{obs}(t)-u(t)
$$
