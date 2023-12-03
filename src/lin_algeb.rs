use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    pub matrix: [[T; C]; R],
    pub row_l: usize,
    pub col_l: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector<T, const C: usize> {
    pub vector: [T; C],
    pub length: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumSolveParam<T> {
    pub eps: T,
    pub nloop: usize,
    pub dis_rate: T,
}

pub trait MatrixOperation<T, const R: usize, const C: usize> {
    fn transpose(&self) -> Matrix<T, C, R>;
}

pub trait DiagMatrixOperation<T, const R: usize> {
    fn transpose_d(&self) -> Matrix<T, R, R>;
    fn eye() -> Matrix<T, R, R>;
    fn diag(num: T) -> Matrix<T, R, R>;
}

impl<T, const R: usize, const C: usize> MatrixOperation<T, R, C> for Matrix<T, R, C>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + num_traits::One
        + Copy
        + Default,
{
    fn transpose(&self) -> Matrix<T, C, R> {
        Matrix {
            matrix: lin_algeb::transpose_matrix(self.matrix),
            row_l: self.col_l,
            col_l: self.row_l,
        }
    }
}

impl<T, const R: usize> DiagMatrixOperation<T, R> for Matrix<T, R, R>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + AddAssign
        + num_traits::One
        + PartialEq
        + PartialOrd
        + Copy
        + Default,
{
    fn transpose_d(&self) -> Matrix<T, R, R> {
        Matrix {
            matrix: lin_algeb::transpose_matrix(self.matrix),
            row_l: self.col_l,
            col_l: self.row_l,
        }
    }

    fn eye() -> Matrix<T, R, R> {
        Matrix {
            matrix: lin_algeb::eye_matrix::<T, R>(),
            row_l: R,
            col_l: R,
        }
    }

    fn diag(num: T) -> Matrix<T, R, R> {
        Matrix {
            matrix: lin_algeb::diag_matrix::<T, R>(num),
            row_l: R,
            col_l: R,
        }
    }
}

impl<T, const R: usize, const C: usize> Add for Matrix<T, R, C>
where
    T: Add<Output = T> + Copy + Default,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Matrix {
            matrix: lin_algeb::add_matrice(self.matrix, rhs.matrix),
            row_l: self.row_l,
            col_l: self.col_l,
        }
    }
}

impl<T, const R: usize, const C: usize> Sub for Matrix<T, R, C>
where
    T: Sub<Output = T> + Copy + Default,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Matrix {
            matrix: lin_algeb::sub_matrice(self.matrix, rhs.matrix),
            row_l: self.row_l,
            col_l: self.col_l,
        }
    }
}

impl<T, const R: usize, const C: usize> Mul for Matrix<T, R, C>
where
    T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Matrix {
            matrix: lin_algeb::mul_matrice(self.matrix, rhs.matrix),
            row_l: self.row_l,
            col_l: rhs.col_l,
        }
    }
}

impl<T, const R: usize, const C: usize> Neg for Matrix<T, R, C>
where
    T: Neg<Output = T> + Copy + Default,
{
    type Output = Self;
    fn neg(self) -> Self {
        Matrix {
            matrix: lin_algeb::mul_neg_eye(self.matrix),
            row_l: self.row_l,
            col_l: self.col_l,
        }
    }
}

impl<T, const C: usize> Add for Vector<T, C>
where
    T: Add<Output = T> + Copy + Default,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Vector {
            vector: lin_algeb::add_vectors(self.vector, rhs.vector),
            length: self.length,
        }
    }
}

impl<T, const C: usize> Sub for Vector<T, C>
where
    T: Sub<Output = T> + Copy + Default,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Vector {
            vector: lin_algeb::sub_vectors(self.vector, rhs.vector),
            length: self.length,
        }
    }
}

impl<T, const C: usize> Mul for Vector<T, C>
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Vector {
            vector: lin_algeb::mul_vectors(self.vector, rhs.vector),
            length: 1,
        }
    }
}

impl<T, const C: usize> Neg for Vector<T, C>
where
    T: Neg<Output = T> + Copy + Default,
{
    type Output = Self;
    fn neg(self) -> Self {
        Vector {
            vector: lin_algeb::mul_neg(self.vector),
            length: self.length,
        }
    }
}

impl<T, const R: usize, const C: usize> Default for Matrix<T, R, C>
where
    T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
{
    fn default() -> Self {
        Matrix {
            matrix: lin_algeb::default_matrix(),
            row_l: R,
            col_l: C,
        }
    }
}

impl<T, const C: usize> Default for Vector<T, C>
where
    T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
{
    fn default() -> Self {
        Vector {
            vector: lin_algeb::default_vector(),
            length: C,
        }
    }
}

pub mod lin_algeb {
    use crate::lin_algeb::{Matrix, Vector};
    use num_traits::One;
    use std::fmt::Display;
    use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

    use super::NumSolveParam;

    pub fn add_matrice<T, const R: usize, const C: usize>(
        matrix_1: [[T; C]; R],
        matrix_2: [[T; C]; R],
    ) -> [[T; C]; R]
    where
        T: Add<Output = T> + Copy + Default,
    {
        let mut result = [[T::default(); C]; R];

        for i in 0..R {
            for j in 0..C {
                result[i][j] = matrix_1[i][j] + matrix_2[i][j];
            }
        }

        result
    }

    pub fn sub_matrice<T, const R: usize, const C: usize>(
        matrix_1: [[T; C]; R],
        matrix_2: [[T; C]; R],
    ) -> [[T; C]; R]
    where
        T: Sub<Output = T> + Copy + Default,
    {
        let mut result = [[T::default(); C]; R];

        for i in 0..R {
            for j in 0..C {
                result[i][j] = matrix_1[i][j] - matrix_2[i][j];
            }
        }

        result
    }

    pub fn mul_matrice<T, const R1: usize, const C1: usize, const R2: usize, const C2: usize>(
        matrix_1: [[T; C1]; R1],
        matrix_2: [[T; C2]; R2],
    ) -> [[T; C1]; R2]
    where
        T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
    {
        if C1 != R2 {
            panic!(
                "In matrix to matrix multiplication, 
            the number of columns of the lhs matrix (length: {C1}) 
            does not correspond with 
            the number of the rows of rhs matrix (length: {R2})"
            );
        }

        let mut result = [[T::default(); C1]; R2];

        for i2 in 0..R2 {
            for j1 in 0..C1 {
                for k in 0..C1 {
                    result[i2][j1] += matrix_1[i2][k] * matrix_2[k][j1];
                }
            }
        }

        result
    }

    pub fn mul_neg_eye<T, const R1: usize, const C1: usize>(
        matrix_1: [[T; C1]; R1],
    ) -> [[T; C1]; R1]
    where
        T: Neg<Output = T> + Copy + Default,
    {
        let mut result = [[T::default(); C1]; R1];

        for i in 0..R1 {
            for j in 0..C1 {
                result[i][j] = -matrix_1[i][j];
            }
        }

        result
    }

    pub fn add_vectors<T, const C: usize>(vector_1: [T; C], vector_2: [T; C]) -> [T; C]
    where
        T: Add<Output = T> + Copy + Default,
    {
        let mut result = [T::default(); C];

        for i in 0..C {
            result[i] = vector_1[i] + vector_2[i];
        }

        result
    }

    pub fn sub_vectors<T, const C: usize>(vector_1: [T; C], vector_2: [T; C]) -> [T; C]
    where
        T: Sub<Output = T> + Copy + Default,
    {
        let mut result = [T::default(); C];

        for i in 0..C {
            result[i] = vector_1[i] - vector_2[i];
        }

        result
    }

    pub fn mul_vectors<T, const C: usize>(vector_1: [T; C], vector_2: [T; C]) -> [T; C]
    where
        T: Add<Output = T> + Mul<Output = T> + Copy + Default,
    {
        let mut result = [T::default(); C];

        for i in 0..C {
            result[i] = vector_1[i] * vector_2[i];
        }

        result
    }

    pub fn mul_neg<T, const C: usize>(vector_1: [T; C]) -> [T; C]
    where
        T: Neg<Output = T> + Copy + Default,
    {
        let mut result = [T::default(); C];

        for i in 0..C {
            result[i] = -vector_1[i];
        }

        result
    }

    pub fn default_matrix<T, const R: usize, const C: usize>() -> [[T; C]; R]
    where
        T: Copy + Default,
    {
        let result = [[T::default(); C]; R];
        result
    }

    pub fn eye_matrix<T, const R: usize>() -> [[T; R]; R]
    where
        T: num_traits::One + Copy + Default,
    {
        let mut result = [[T::default(); R]; R];
        for i in 0..R {
            result[i][i] = T::one();
        }
        result
    }

    pub fn diag_matrix<T, const R: usize>(num: T) -> [[T; R]; R]
    where
        T: Copy + Default,
    {
        let mut result = [[num; R]; R];
        for i in 0..R {
            result[i][i] = num;
        }
        result
    }

    pub fn transpose_matrix<T, const R: usize, const C: usize>(matrix: [[T; C]; R]) -> [[T; R]; C]
    where
        T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
    {
        let mut result = [[T::default(); R]; C];
        for i in 0..R {
            for j in 0..C {
                result[j][i] = matrix[i][j];
            }
        }

        result
    }

    pub fn default_vector<T, const C: usize>() -> [T; C]
    where
        T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
    {
        let result = [T::default(); C];
        result
    }

    pub fn mul_matrix_vector<T, const R: usize, const C: usize>(
        matrix: Matrix<T, R, C>,
        vector: Vector<T, C>,
    ) -> Vector<T, R>
    where
        T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
    {
        let mut result = Vector::<T, R>::default();

        for row in 0..R {
            for col in 0..C {
                result.vector[row] += matrix.matrix[row][col] * vector.vector[col];
            }
        }

        result
    }

    pub fn norm_2d<T, const C: usize>(vector: Vector<T, C>) -> T
    where
        T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
    {
        let mut result = T::default();

        for i in 0..C {
            result += vector.vector[i] * vector.vector[i];
        }

        result
    }

    pub fn abs<T>(num: T) -> T
    where
        T: Neg<Output = T> + Copy + Default + PartialOrd,
    {
        let mut result = num;
        if num < T::default() {
            result = -num;
        }
        result
    }

    pub fn swap_matrix_row<T, const R: usize, const C: usize>(
        matrix: &mut [[T; C]; R],
        i_row: usize,
        j_row: usize,
    ) where
        T: Copy + Default,
    {
        for k in 0..R {
            let tmp = matrix[i_row][k];
            matrix[i_row][k] = matrix[j_row][k];
            matrix[j_row][k] = tmp;
        }
    }

    fn pivot<T, const R: usize>(matrix: &mut [[T; R]; R], param: NumSolveParam<T>) -> [[T; R]; R]
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + AddAssign
            + One
            + Copy
            + PartialEq
            + PartialOrd
            + Default,
    {
        let mut pivot = eye_matrix::<T, R>();

        for k in 0..R {
            let mut max = abs(matrix[k][k]);
            let mut current_pivot = k;
            for i in (k + 1)..R {
                let pv_cand = abs(matrix[i][k]);
                if pv_cand > max {
                    max = pv_cand;
                    current_pivot = i;
                }
            }

            // not regular matrix
            if max < param.eps {
                panic!(
                    "This matrix does not have any proper pivot at {}th row 
                because it is not regular.",
                    k
                );
            }

            if current_pivot != k {
                swap_matrix_row(&mut pivot, k, current_pivot);
                swap_matrix_row(matrix, k, current_pivot);
            }
        }

        pivot
    }

    // based on Crout's algorithm
    pub fn lu_decomposition<T, const R: usize>(
        matrix: &mut Matrix<T, R, R>,
        param: NumSolveParam<T>,
    ) -> (Matrix<T, R, R>, Matrix<T, R, R>, Matrix<T, R, R>)
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + AddAssign
            + One
            + PartialEq
            + PartialOrd
            + Copy
            + Default,
    {
        let mut result_l = Matrix {
            matrix: [[T::default(); R]; R],
            row_l: R,
            col_l: R,
        };
        let mut result_u = Matrix {
            matrix: [[T::default(); R]; R],
            row_l: R,
            col_l: R,
        };
        let pivot = Matrix {
            matrix: pivot(&mut matrix.matrix, param),
            row_l: R,
            col_l: R,
        };

        for i in 0..R {
            result_l.matrix[i][i] = T::one();
        }

        for i in 0..R {
            for j in 0..R {
                let mut sum = T::default();
                for k in 0..j {
                    sum += result_l.matrix[i][k] * result_u.matrix[k][j];
                }
                if i <= j {
                    result_u.matrix[i][j] = matrix.matrix[i][j] - sum;
                } else {
                    result_l.matrix[i][j] = (matrix.matrix[i][j] - sum) / result_u.matrix[j][j];
                }
                if i == j && abs(result_u.matrix[j][j]) < param.eps {
                    panic!("This matrix cannot be decomposed bacause it is singular.");
                }
            }
        }

        (result_l, result_u, pivot)
    }

    pub fn show_matrix<T, const R: usize, const C: usize>(matrix: Matrix<T, R, C>)
    where
        T: Copy + Default + Display,
    {
        for row in 0..R {
            for col in 0..C {
                print!("{:.5}, ", matrix.matrix[row][col]);
            }
            print!("\n");
        }
        print!("\n");
    }

    pub fn take_schur_prod_matrix<T, const R: usize, const C: usize>(
        matrix_1: &Matrix<T, R, C>,
        matrix_2: &Matrix<T, R, C>,
    ) -> Matrix<T, R, C>
    where
        T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
    {
        let mut result = Matrix::<T, R, C>::default();

        for row in 0..R {
            for col in 0..C {
                result.matrix[row][col] = matrix_1.matrix[row][col] * matrix_2.matrix[row][col];
            }
        }

        result
    }

    pub fn take_schur_prod_vector<T, const C: usize>(
        vector_1: &Vector<T, C>,
        vector_2: &Vector<T, C>,
    ) -> Vector<T, C>
    where
        T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
    {
        let mut result = Vector::<T, C>::default();

        for i in 0..C {
            result.vector[i] = vector_1.vector[i] * vector_2.vector[i];
        }

        result
    }

    pub fn convert_vector_to_diagmatrix<T, const C: usize>(vector: &Vector<T, C>) -> Matrix<T, C, C>
    where
        T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy + Default,
    {
        let mut result = Matrix::<T, C, C>::default();

        for i in 0..C {
            result.matrix[i][i] = vector.vector[i];
        }

        result
    }

    pub fn invert_vector_elementwise<T, const C: usize>(vector: &Vector<T, C>) -> Vector<T, C>
    where
        T: Add<Output = T> + Div<Output = T> + AddAssign + Copy + Default + One,
    {
        let mut result = Vector::<T, C>::default();

        for i in 0..C {
            result.vector[i] = T::one() / vector.vector[i];
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::lin_algeb;
    use super::NumSolveParam;
    use crate::lin_algeb::lin_algeb::*;
    use crate::lin_algeb::Matrix;

    #[test]
    fn test_lu_decomposition() {
        let mut m_1 = Matrix {
            matrix: [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [3.0, 1.0, 0.0]],
            row_l: 3,
            col_l: 3,
        };
        let l_t = Matrix {
            matrix: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0 / 3.0, -1.0 / 3.0, 1.0],
            ],
            row_l: 3,
            col_l: 3,
        };
        let u_t = Matrix {
            matrix: [[3.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 11.0 / 3.0]],
            row_l: 3,
            col_l: 3,
        };
        let pivot_t = Matrix {
            matrix: [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            row_l: 3,
            col_l: 3,
        };
        let param = NumSolveParam {
            eps: 1e-4,
            nloop: 100,
            dis_rate: 1.0,
        };

        let (l, u, pivot) = lu_decomposition(&mut m_1, param);
        assert_eq!(l, l_t);
        assert_eq!(u, u_t);
        assert_eq!(pivot, pivot_t);
    }

    #[test]
    fn test_take_schur_prod_matrix() {
        let a = Matrix {
            matrix: [[1.0, 2.0, -1.0], [2.0, -3.0, -1.0], [3.0, 2.0, 1.0]],
            row_l: 3,
            col_l: 3,
        };
        let b = Matrix {
            matrix: [[1.0, 0.0, -1.0], [2.0, 3.0, -1.0], [1.0, 0.0, 1.0]],
            row_l: 3,
            col_l: 3,
        };

        let c = lin_algeb::take_schur_prod_matrix(&a, &b);

        let c_t = Matrix {
            matrix: [[1.0, 0.0, 1.0], [4.0, -9.0, 1.0], [3.0, 0.0, 1.0]],
            row_l: 3,
            col_l: 3,
        };

        assert_eq!(c, c_t);
    }
}
