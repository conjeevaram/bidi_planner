from __future__ import annotations

TITLE_A = r"$A$ · conventional · directed $\theta^*$ · $\rho=2$"
TITLE_B = r"$B$ · bidirectional · axis goal · $\rho=1$"

XLABEL_RHO = r"$J^*(A) - J^*(\rho\!=\!1)$"
YLABEL_GOAL = r"$J^*(\rho\!=\!1) - J^*(B)$"
CBAR_TOTAL = r"$J^*(A) - J^*(B)$"

LEGEND_A = r"$A$ conventional vehicle, directed $\theta^*$, reverse penalty $\rho=2$"
LEGEND_B = (
    r"$B$ bidirectional vehicle, "
    r"axis goal $\{\theta^*, \theta^*\!+\!\pi\}$, $\rho=1$"
)
LEGEND_J = (
    r"$J^*$ = optimal Reeds-Shepp path cost "
    r"(sum of arc lengths; each reverse segment $\times\,\rho$)"
)

OVERVIEW_TITLE = r"Nonholonomic planning: conventional vs bidirectional ($SE(2)$, Reeds-Shepp)"
VIDEO_METADATA_TITLE = "Reeds-Shepp: conventional vs bidirectional"
MARK_AT_GOAL = r"$\checkmark$"

COLOR_A = "#2563eb"
COLOR_B = "#16a34a"
COLOR_REVERSE = "crimson"


def decomposition_line(delta_rho: float, delta_theta: float) -> str:
    total = delta_rho + delta_theta
    return (
        rf"$\Delta_\rho={delta_rho:.2f}$ · "
        rf"$\Delta_\theta={delta_theta:.2f}$ · "
        rf"$\Delta J^*={total:.2f}$"
    )
