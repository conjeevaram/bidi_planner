use pyo3::prelude::*;
use reeds_shepp_lib::{get_all_paths, Gear, PathElement, Pose, Steering};

fn pose_from(x: f64, y: f64, theta_rad: f64, turning_radius: f64) -> Pose {
    Pose {
        x: x / turning_radius,
        y: y / turning_radius,
        theta_degree: theta_rad.to_degrees(),
    }
}

fn element_to_tuple(e: &PathElement, turning_radius: f64) -> (char, char, f64) {
    let steering = match e.steering {
        Steering::Left => 'L',
        Steering::Right => 'R',
        Steering::Straight => 'S',
    };
    let gear = match e.gear {
        Gear::Forward => 'F',
        Gear::Backwards => 'B',
    };
    (steering, gear, e.param.abs() * turning_radius)
}

#[pyfunction]
#[pyo3(signature = (sx, sy, st, gx, gy, gt, turning_radius=1.0))]
fn get_all_paths_py(
    sx: f64,
    sy: f64,
    st: f64,
    gx: f64,
    gy: f64,
    gt: f64,
    turning_radius: f64,
) -> PyResult<Vec<Vec<(char, char, f64)>>> {
    let start = pose_from(sx, sy, st, turning_radius);
    let end = pose_from(gx, gy, gt, turning_radius);
    Ok(get_all_paths(start, end)
        .iter()
        .map(|path| path.iter().map(|e| element_to_tuple(e, turning_radius)).collect())
        .collect())
}

#[pymodule]
fn reeds_shepp_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_all_paths_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
