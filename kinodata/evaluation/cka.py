# All taken from https://github.com/RistoAle97/centered-kernel-alignment/tree/main/src/ckatorch
# MIT License
#
# Copyright (c) 2024 Alessandro Ristori
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Literal

import torch


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    """Computes the Gram (kernel) matrix for a linear kernel.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x (torch.Tensor): tensor of shape (n, m).

    Returns:
        torch.Tensor: a Gram matrix which is a tensor of shape (n, n).
    """
    return torch.mm(x, x.T)


def rbf_kernel(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Computes the Gram (kernel) matrix for an RBF kernel.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x (torch.Tensor): tensor of shape (n, m).
        threshold (float): fraction of median Euclidean distance to use as RBF kernel bandwidth (default=1.0).

    Returns:
        torch.Tensor: a Gram matrix which is a tensor of shape (n, n).
    """
    dot_products = torch.mm(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram_matrix(
    gram_matrix: torch.Tensor, unbiased: bool = False
) -> torch.Tensor:
    """Centers a given Gram matrix.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        gram_matrix (torch.Tensor): tensor of shape (n, n).
        unbiased (bool): whether to use the unbiased version of the centering (default=False).

    Returns:
        torch.Tensor: the centered version of the given Gram matrix.
    """
    if not torch.allclose(gram_matrix, gram_matrix.T):
        raise ValueError("The given matrix must be symmetric.")

    gram_matrix = gram_matrix.detach().clone()
    if unbiased:
        n = gram_matrix.shape[0]
        gram_matrix.fill_diagonal_(0)
        means = torch.sum(gram_matrix, dim=0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]
        gram_matrix.fill_diagonal_(0)
    else:
        means = torch.mean(gram_matrix, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]

    return gram_matrix


def hsic0(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the Hilbert-Schmidt Independence Criterion on two given Gram matrices.

    Args:
        gram_x (torch.Tensor): Gram matrix of shape (n, n), this is equivalent to K from the original paper.
        gram_y (torch.Tensor): Gram matrix of shape (n, n), this is equivalent to L from the original paper.

    Returns:
        torch.Tensor: a tensor with the Hilbert-Schmidt Independence Criterion values.

    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` are not symmetric.
    """
    if not torch.allclose(gram_x, gram_x.T) and not torch.allclose(gram_y, gram_y.T):
        raise ValueError("The given matrices must be symmetric.")

    # Build the identity matrix
    n = gram_x.shape[0]
    identity = torch.eye(n, n, dtype=gram_x.dtype, device=gram_x.device)

    # Build the centering matrix
    h = identity - torch.ones(n, n, dtype=gram_x.dtype, device=gram_x.device) / n

    # Compute k * h and l * h
    kh = torch.mm(gram_x, h)
    lh = torch.mm(gram_y, h)

    # Compute the trace of the product kh * lh
    trace = torch.trace(kh.mm(lh))
    return trace / (n - 1) ** 2


def partial_hsic0_1(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
    gram_control: torch.Tensor,
    reg: float = 1e-5,
):
    """Compute the controlled Hilbert-Schmidt Independence Criterion (Partial HSIC).

    Args:
        gram_x (torch.Tensor): (n, n) Gram matrix of X.
        gram_y (torch.Tensor): (n, n) Gram matrix of Y.
        gram_control (torch.Tensor): (n, n) Gram matrix of control variable Z.
        reg (float): regularization term for inversion stability.

    Returns:
        torch.Tensor: Partial HSIC scalar value.
    """
    if (
        not torch.allclose(gram_x, gram_x.T)
        or not torch.allclose(gram_y, gram_y.T)
        or not torch.allclose(gram_control, gram_control.T)
    ):
        raise ValueError("The given matrices must be symmetric.")

    n = gram_x.shape[0]
    device = gram_x.device
    dtype = gram_x.dtype

    # Centering matrix
    h = (
        torch.eye(n, dtype=dtype, device=device)
        - torch.ones(n, n, dtype=dtype, device=device) / n
    )

    # Center the kernels
    K = h @ gram_x @ h
    L = h @ gram_y @ h
    C = h @ gram_control @ h

    # Projection matrix onto the space spanned by C (with regularization)
    reg_eye = reg * torch.eye(n, dtype=dtype, device=device)
    C_inv = torch.linalg.inv(C + reg_eye)
    P_C = C @ C_inv

    # Residualize K and L
    K_resid = (torch.eye(n, device=device) - P_C) @ K
    L_resid = (torch.eye(n, device=device) - P_C) @ L

    # Partial HSIC = trace of the product of residualized kernels
    hsic_partial = torch.trace(K_resid @ L_resid) / ((n - 1) ** 2)

    return hsic_partial


def partial_hsic0(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
    gram_control: torch.Tensor,
    reg: float = 1e-5,
) -> torch.Tensor:
    """
    Compute the Partial Hilbert-Schmidt Independence Criterion (Partial HSIC).

    Args:
        gram_x (torch.Tensor): (n, n) Gram matrix of X.
        gram_y (torch.Tensor): (n, n) Gram matrix of Y.
        gram_control (torch.Tensor): (n, n) Gram matrix of control variable Z.
        reg (float): Regularization parameter for numerical stability.

    Returns:
        torch.Tensor: Partial HSIC value (scalar).
    """
    if (
        not torch.allclose(gram_x, gram_x.T)
        or not torch.allclose(gram_y, gram_y.T)
        or not torch.allclose(gram_control, gram_control.T)
    ):
        raise ValueError("All input Gram matrices must be symmetric.")

    n = gram_x.shape[0]
    device = gram_x.device
    dtype = gram_x.dtype

    I = torch.eye(n, dtype=dtype, device=device)
    ones = torch.ones(n, n, dtype=dtype, device=device)
    H = I - ones / n  # Centering matrix

    # Regularized projection matrix onto the RKHS span of control variable
    reg_eye = reg * I
    P = gram_control @ torch.linalg.solve(gram_control + reg_eye, I)

    # Residualize X and Y with respect to control
    K_resid = (I - P) @ gram_x @ (I - P)
    L_resid = (I - P) @ gram_y @ (I - P)

    # Center the residualized Gram matrices
    K_resid = H @ K_resid @ H
    L_resid = H @ L_resid @ H

    # Compute Partial HSIC as normalized trace of residual product
    hsic_partial = torch.trace(K_resid @ L_resid) / ((n - 1) ** 2)

    return hsic_partial


def cka_base(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: Literal["linear", "rbf"] = "linear",
    unbiased: bool = False,
    threshold: float = 1.0,
    method: Literal["fro_norm", "hsic"] = "hsic",
) -> torch.Tensor:
    """Computes the Centered Kernel Alignment (CKA) between two given matrices.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x (torch.Tensor): tensor of shape (n, j).
        y (torch.Tensor): tensor of shape (n, k).
        kernel (Literal["linear", "rbf"]): the kernel used to compute the Gram matrices, must be "linear" or "rbf"
            (default="linear").
        unbiased (bool): whether to use the unbiased version of CKA (default=False).
        threshold (float): the threshold used by the RBF kernel (default=1.0).
        method (Literal["fro_norm", "hsic"]): the method used to compute the CKA value, must be "fro_norm"
            (Frobenius norm) or "hsic" (Hilbert-Schmidt Independence Criterion). Note that the choice does not
            influence the output (default="fro_norm").

    Returns:
        torch.Tensor: a float tensor in [0, 1] that is the CKA value between the two given matrices.

    Raises:
        ValueError: if ``kernel`` is not "linear" or "rbf" or if ``method`` is not "fro_norm" or "hsic".
    """
    if kernel not in ["linear", "rbf"]:
        raise ValueError("The chosen kernel must be either 'linear' or 'rbf'.")

    if method not in ["hsic", "fro_norm"]:
        raise ValueError("The chosen method must be either 'hsic' or 'fro_norm'.")

    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y

    # Build the Gram matrices by applying the kernel
    gram_x = linear_kernel(x) if kernel == "linear" else rbf_kernel(x, threshold)
    gram_y = linear_kernel(y) if kernel == "linear" else rbf_kernel(y, threshold)

    # Compute CKA by either using HSIC or the Frobenius norm
    if method == "hsic":
        hsic_xy = hsic0(gram_x, gram_y)
        hsic_xx = hsic0(gram_x, gram_x)
        hsic_yy = hsic0(gram_y, gram_y)
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    else:
        gram_x = center_gram_matrix(gram_x, unbiased)
        gram_y = center_gram_matrix(gram_y, unbiased)
        norm_xy = gram_x.ravel().dot(gram_y.ravel())
        norm_xx = torch.linalg.norm(gram_x, ord="fro")
        norm_yy = torch.linalg.norm(gram_y, ord="fro")
        cka = norm_xy / (norm_xx * norm_yy)

    return cka


def partial_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    control: torch.Tensor,
    kernel: Literal["linear", "rbf"] = "linear",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Computes the controlled Centered Kernel Alignment (CKA) between two given matrices."""

    if kernel not in ["linear", "rbf"]:
        raise ValueError("The chosen kernel must be either 'linear' or 'rbf'.")

    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y
    control = (
        control.type(torch.float64) if not control.dtype == torch.float64 else control
    )

    # Build the Gram matrices by applying the kernel
    gram_x = linear_kernel(x) if kernel == "linear" else rbf_kernel(x, threshold)
    gram_y = linear_kernel(y) if kernel == "linear" else rbf_kernel(y, threshold)
    gram_control = (
        linear_kernel(control) if kernel == "linear" else rbf_kernel(control, threshold)
    )

    # Compute CKA by using HSIC
    phsic_xy = partial_hsic0(gram_x, gram_y, gram_control)
    phsic_xx = partial_hsic0(gram_x, gram_x, gram_control)
    phsic_yy = partial_hsic0(gram_y, gram_y, gram_control)
    cka = phsic_xy / torch.sqrt(phsic_xx * phsic_yy)

    return cka


def soft_partial_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    control: torch.Tensor,
    kernel: Literal["linear", "rbf"] = "linear",
    threshold: float = 1.0,
) -> torch.Tensor:
    if kernel not in ["linear", "rbf"]:
        raise ValueError("The chosen kernel must be either 'linear' or 'rbf'.")

    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y
    control = (
        control.type(torch.float64) if not control.dtype == torch.float64 else control
    )

    # Build the Gram matrices by applying the kernel
    gram_x = linear_kernel(x) if kernel == "linear" else rbf_kernel(x, threshold)
    gram_y = linear_kernel(y) if kernel == "linear" else rbf_kernel(y, threshold)
    gram_control = (
        linear_kernel(control) if kernel == "linear" else rbf_kernel(control, threshold)
    )
    gram_y = (2 * gram_y * (gram_y - gram_control).abs()) / (
        gram_y.abs() + gram_control.abs() + 1e-8
    )
    hsic_xy = hsic0(gram_x, gram_y)
    hsic_xx = hsic0(gram_x, gram_x)
    hsic_yy = hsic0(gram_y, gram_y)
    cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)

    return cka
