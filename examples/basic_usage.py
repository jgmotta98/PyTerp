import numpy as np
import pandas as pd
import pyterp as pt
import time


def create_fake_dataframes(num_rows: int = 10) -> None:
    data = {
        'X': np.random.rand(num_rows) * 10,
        'Y': np.random.rand(num_rows) * 10,
        'Z': np.random.rand(num_rows) * 5,
        'RANDOM_VARIABLE': np.random.rand(num_rows) * 10 + 5
    }

    df = pd.DataFrame(data)
    df = df.sort_values(by=['X', 'Y', 'Z']).reset_index(drop=True)
    df_rounded = df.round(2)

    return df_rounded


def interpolated_coordinates(df: pd.DataFrame, step_x: float = 0.1, step_y: float = 0.1, 
                             step_z: float = 0.1) -> np.ndarray[float]:
    x_new = np.arange(df['X'].min(), df['X'].max(), step_x)
    y_new = np.arange(df['Y'].min(), df['Y'].max(), step_y)
    z_new = np.arange(df['Z'].min(), df['Z'].max(), step_z)

    X_grid, Y_grid, Z_grid = np.meshgrid(x_new, y_new, z_new, indexing='ij')

    target_points = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T

    return target_points


def main() -> None:
    df = create_fake_dataframes()

    source_points = df[['X', 'Y', 'Z']].to_numpy(dtype=np.float64)
    source_values = df['RANDOM_VARIABLE'].to_numpy(dtype=np.float64)
    intp_grid = interpolated_coordinates(df)

    start = time.perf_counter()
    interpolated_values = pt.interpolate(
        source_points=source_points,
        source_values=source_values,
        target_points=intp_grid,
        k_neighbors=10,
        power=2
    )
    end = time.perf_counter() - start

    final_df = pd.DataFrame(intp_grid, columns=['X', 'Y', 'Z'])
    final_df['RANDOM_INTERPOLATED_VARIABLE'] = interpolated_values

    print(df)
    print(final_df)

    print(f"Interpolation took {(end * 1000):.0f} ms to be made!")


if __name__ == "__main__":
    main()