Automatic Batching
==================

In this topic, we will discuss the automatic batching mechanism of N3LDG++ using Transformers' self-attention's forward pass as an example. To simplify the illustration, it will not cover multi-head attention, but the method we will discuss can generalize to any operation. 

The Self-attention's Forward Pass Example
-----------------------------------------

Given a mini-batch containing matrices :math:`\{X_i\}_{i=1}^b` satisfying :math:`row(X_i) = d` (but :math:`col(X_i)` are commonly not equal), where :math:`row(X_i)` and :math:`col(X_i)` returns the row and column number of :math:`X_i`, respectively. We can pass them to self-attention as follows:

.. math::

    \begin{align}
        Q_i = W_Q X_i\tag{1}\\
        K_i = W_K X_i\tag{2}\\
        V_i = W_V X_i\tag{2}\\
        S_i = {K_i}^T Q_i\tag{3}\\
        {S_i}'= \frac{1}{\sqrt{d}} S_i\tag{4}\\
        A_i = softmax({S_i}')\tag{5}\\
        Y_i = A_i V_i\tag{6}
    \end{align}

Executing the same formula in batch can generally speed up computation, especially on the GPU. To this end, following DyNet, N3LDG++ maps every operation into a signature, where operations with the same signature mean that they should be computed in batch. In the following, we will discuss how N3LDG++ executes the above formulas one by one.

Linear Transformation
---------------------

Formula (1) ~ (3) are all linear transformations,  and taking Formular (1) as the example, we can first merge :math:`X_1, X_2, ... , X_b` into a single matrix :math:`\bigl[ \begin{smallmatrix}X_1 & X_2 & ... & X_b\end{smallmatrix} \bigr]`, and then compute :math:`\bigl[ \begin{smallmatrix}Q_1 & Q_2 & ... & Q_b\end{smallmatrix} \bigr] = W_Q \bigl[ \begin{smallmatrix}X_1 & X_2 & ... & X_b\end{smallmatrix} \bigr]`. Finally, we split the matrix :math:`\bigl[ \begin{smallmatrix}Q_1 & Q_2 & ... & Q_b\end{smallmatrix} \bigr]` into matrices :math:`Q_1, Q_2, ... , Q_b`.

From the above computation process, we find that linear transformations with the same parameter can be executed in batch. Thus we set the signature to :code:`"linear-" + addr(W)`.

Matrix Multiplication after Transposing the Former
--------------------------------------------------
