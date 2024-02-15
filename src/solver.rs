use crate::lin_algeb::lin_algeb::*;
use crate::lin_algeb::*;

/// solver for linear programming
pub mod lin_solver {
    use super::NumSolveParam;
    use crate::lin_algeb::lin_algeb::*;
    use crate::lin_algeb::{DiagMatrixOperation, Matrix, MatrixOperation, Vector};
    use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
    use std::task::Context;

    fn primal_dual_interior_point_method_for_lin_mpc<T, const QDIM: usize, const SLACKS: usize>(
        quad_cost_mat: Matrix<T, QDIM, QDIM>,
        lin_cost_mat: Matrix<T, QDIM, QDIM>,
        ineq_constraints_matrix: Matrix<T, SLACKS, QDIM>,
        solve_params: NumSolveParam<T>,
    ) where
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
        let mut res_u = Vector::<T, QDIM>::default();
        let mut res_slack = Vector::<T, SLACKS>::default();
        let mut dual_gap = 1e5;

        while dual_gap > solve_params.eps {}
    }

    pub fn lin_mpc<T, const STATES: usize, const INPUTS: usize, const TF: usize>(
        initial_states: Vector<T, STATES>,
        // control_input: Vector<T, INPUTS>,
        a_mat: Matrix<T, STATES, STATES>,
        b_mat: Matrix<T, STATES, INPUTS>,
        q_mat: Matrix<T, STATES, STATES>,
        q_f_mat: Matrix<T, STATES, STATES>,
        r_mat: Matrix<T, INPUTS, INPUTS>,
        state_min: Option<Vector<T, STATES>>,
        state_max: Option<Vector<T, STATES>>,
        input_min: Option<Vector<T, INPUTS>>,
        input_max: Option<Vector<T, INPUTS>>,
        horizon: usize,
        t_s: usize,
        ref_state_trajectory: Option<Vector<T, TF>>,
        ref_input_trajectory: Option<Vector<T, TF>>,
    ) where
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
        let mut input_res = Vector::<T, STATES>::default();
        let mut current_state = initial_states;
        for step in 0..TF {}
    }
}
