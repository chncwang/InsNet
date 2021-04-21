Automatic Batching
==================

In this topic, we will discuss the automatic batching mechanism of N3LDG++ using Transformers' self-attention's forward pass as an example. To simplify the illustration, it will not cover multi-head attention, but the method we will discuss can generalize to any operator. 

The Self-attention's Forward Pass Example
-----------------------------------------

Given a mini-batch containing matrices :math:`\{X_i\}_{i=1}^b` satisfying :math:`row(X_i) = d` (but :math:`col(X_i)` **are commonly not equal** since we do not use paddings to align shapes), where :math:`row(X_i)` and :math:`col(X_i)` returns the row and column number of :math:`X_i`, respectively. We can pass them to self-attention as follows:

.. math::

    \begin{align}
        Q_i = W_Q X_i\tag{1}\\
        K_i = W_K X_i\tag{2}\\
        V_i = W_V X_i\tag{3}\\
        S_i = {K_i}^T Q_i\tag{4}\\
        {S_i}'= \frac{1}{\sqrt{d}} S_i\tag{5}\\
        A_i = softmax({S_i}')\tag{6}\\
        Y_i = A_i V_i\tag{7}
    \end{align}

Executing the same formula in batch can generally speed up computation, especially on the GPU. To this end, following DyNet, N3LDG++ maps every operator into a signature, where operators with the same signature mean that they should be computed in batch. In the following, we will discuss how N3LDG++ executes the above formulas one by one.

:math:`Y = W X`
---------------------

Formula (1) ~ (3) are all linear transformations,  and taking Formular (1) as the example, we can first merge :math:`X_1, X_2, ... , X_b` into a single matrix :math:`\bigl[ \begin{smallmatrix}X_1 & X_2 & ... & X_b\end{smallmatrix} \bigr]`, and then compute :math:`\bigl[ \begin{smallmatrix}Q_1 & Q_2 & ... & Q_b\end{smallmatrix} \bigr] = W_Q \bigl[ \begin{smallmatrix}X_1 & X_2 & ... & X_b\end{smallmatrix} \bigr]`. Finally, we split the matrix :math:`\bigl[ \begin{smallmatrix}Q_1 & Q_2 & ... & Q_b\end{smallmatrix} \bigr]` into matrices :math:`Q_1, Q_2, ... , Q_b`.

From the above computation process, we find that linear transformations with the same parameter can be executed in batch. Thus we set the signature to :code:`"linear-" + addr(W)`.

:math:`Y = A^T B`
-------------------------

One way to implement formula (4) is to divide it into two operators, i.e., matrix transposition and matrix multiplication, which would help keep public APIs fine-grained and orthogonal. However, considering the additional data transfer of the former matrix it would cause, we still implement it as one operator.

Then we need to determine that when the input matrices meet what condition, the operators can be executed in batch. Still taking formula (4) as the example, obviously, the condition should not be that the shapes of :math:`\{K_i\}_{i=1}^b` and :math:`\{Q_i\}_{i=1}^b` are equal, respectively, for :math:`col(Q_i)` and :math:`col(K_i)` are not equal in a mini-batch. Thus we set the signature to :code:`"transpose-mul-" + to_string(row(A))`

:math:`Y = \alpha X`
--------------------------------------

Taking formula (5) as the example, since the sizes of :math:`\{S_i\}_{i=1}^b` are not equal in the mini-batch, we define the signature as :code:`"factor"`, i.e., N3LDG++ executes this type of operators in batch regardless of the input sizes.

More generally, N3LDG++ executes all single-input operators :math:`Y = F(X)` satisfying :math:`y_i = f(x_i), 0 < i < size(X)`, e.g., :code:`dropout`, :code:`tanh` and :code:`relu` in batch regardless of their sizes.

:math:`Y = softmax(X)`
--------------------------

Similarly, we define softmax's signature as :code:`"softmax"`, which means N3LDG++ executes all softmax operators in batch.

:math:`Y = A B`
------------------

To properly execute formula (7) in batch, we define the matrix multiplication's signature as :code:`"mul-" + row(A)`

The Necessity of Exploiting Model Design Bias
------------------------------------------------

One may concern that shall we define these general-purpose operators' signatures to adapt self-attention? More generally, shall we exploit model design bias?

Our answer is "Yes" because the automatic batching problem is only tractable when exploiting model design bias. For example, recall how N3LDG++ batch :math:`Y = W X` and we can realize that the efficiency of :math:`\bigl[ \begin{smallmatrix}Y_1 & Y_2 & ... & Y_b\end{smallmatrix} \bigr] = W \bigl[ \begin{smallmatrix}X_1 & X_2 & ... & X_b\end{smallmatrix} \bigr]` is based on the assumption that :math:`W` is shared in a mini-batch. Otherwise, why not try :math:`Y = \bigl[ \begin{smallmatrix}W_1 W_2 & ... & W_N\end{smallmatrix} \bigr]^T X`?
