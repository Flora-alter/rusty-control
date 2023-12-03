use crate::lin_algeb::lin_algeb::*;
use crate::lin_algeb::*;

/// solver for linear programming
pub mod lin_solver {
    use super::NumSolveParam;
    use crate::lin_algeb::lin_algeb::*;
    use crate::lin_algeb::{DiagMatrixOperation, Matrix, MatrixOperation, Vector};
    use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

    pub fn karmarker<T, const RINEQ: usize, const CINEQ: usize>(
        obj_variables: Vector<T, CINEQ>,
        obj_vector: Vector<T, CINEQ>,
        ineq_constraints_matrix: Matrix<T, RINEQ, CINEQ>,
        ineq_constraints_vector: Vector<T, RINEQ>,
        param: NumSolveParam<T>,
    ) -> Vector<T, CINEQ>
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Neg<Output = T>
            + Div<Output = T>
            + num_traits::One
            + AddAssign
            + Copy
            + Default,
    {
        let mut res = Vector::<T, CINEQ>::default();
        for t in 0..param.nloop {
            let vk =
                ineq_constraints_vector - mul_matrix_vector(ineq_constraints_matrix, obj_variables);
            let vk_sq = invert_vector_elementwise(&take_schur_prod_vector(&vk, &vk));
            let diag = convert_vector_to_diagmatrix(&vk_sq);
            let gmat = ineq_constraints_matrix.transpose() * diag * ineq_constraints_matrix;
        }
        res
    }
}
