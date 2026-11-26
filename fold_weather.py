"""Fold-local versions of v2's three day-of-year weather anomalies."""

from datetime import date

import numpy as np
import pyarrow.parquet as pq

from build_grid_weather import DAILY, GRID, WEIGHTS, require_pin, table_hash
from build_model_table_v2 import (
    ANOMALY_FIELDS, BASE_FIELDS, LOOKBACK, NORMAL_RADIUS, april_weather, day_of_year, warmup_inputs,
)


class FoldWeather:
    def __init__(self, years, features, model_metadata):
        self.columns = [features.index(f"{field}_anomaly_doy7") for field in ANOMALY_FIELDS]
        self.current_columns = [features.index(field) for field in ANOMALY_FIELDS]
        ids = pq.read_table(GRID, columns=['cell_id'])['cell_id'].to_numpy()
        if model_metadata.get(b'grid_sha256') != table_hash(GRID).encode():
            raise ValueError('Model table and fold-normal weather use different grids')
        donors, matrix, cache = warmup_inputs(ids)
        self.doys = np.stack([day_of_year(np.arange(f'{y}-04-01', f'{y}-11-01', dtype='datetime64[D]')) for y in years])
        self.day_count = self.doys.shape[1]
        self.cells = len(ids)
        self.first = self.doys[:, 0]
        self.last = self.doys[:, -1]
        self.lower = self.doys[:, LOOKBACK:-1] - NORMAL_RADIUS - 1
        self.upper = self.doys[:, LOOKBACK:-1] + NORMAL_RADIUS
        # Per-year prefixes allow both inner and outer fits to exclude their
        # validation years, and allow every fitting row to exclude its own year.
        self.prefix = np.zeros((len(years), len(ANOMALY_FIELDS), 367, self.cells))
        dates = np.concatenate([np.arange(f'{y}-05-01', f'{y}-11-01', dtype='datetime64[D]') for y in years])
        with pq.ParquetFile(DAILY, pre_buffer=False) as parquet:
            schema = parquet.schema_arrow
            require_pin(schema.metadata, DAILY)
            if schema.metadata.get(b'weights_sha256') != table_hash(WEIGHTS).encode():
                raise ValueError('April warm-up and grid weather use different weights')
            stop = date(years[-1] + 1, 1, 1)
            group_ids = []
            for g in range(parquet.num_row_groups):
                stats = parquet.metadata.row_group(g).column(schema.get_field_index('date')).statistics
                if stats is None or not stats.has_min_max or stats.null_count != 0:
                    raise ValueError(f'Grid weather group {g} lacks date bounds')
                if stats.min >= stop:
                    break
                if stats.max >= stop:
                    raise ValueError(f'Grid weather group {g} crosses the training cutoff')
                group_ids.append(g)
            batches = parquet.iter_batches(row_groups=group_ids, batch_size=self.cells * 8,
                                           columns=['date', 'cell_id', *ANOMALY_FIELDS])
            offset = 0
            season_days = self.day_count - LOOKBACK
            for batch in batches:
                days, partial = divmod(batch.num_rows, self.cells)
                end = offset + days
                if partial or end > len(dates) or any(c.null_count for c in batch.columns):
                    raise ValueError('Invalid fold-normal source batch')
                if not ((batch.column('date').to_numpy(zero_copy_only=False).reshape(days, -1) == dates[offset:end, None]).all()
                        and (batch.column('cell_id').to_numpy().reshape(days, -1) == ids).all()):
                    raise ValueError('Unexpected fold-normal dates or cell IDs')
                values = [batch.column(f).to_numpy().reshape(days, -1) for f in ANOMALY_FIELDS]
                if not all(np.isfinite(v).all() for v in values):
                    raise ValueError('Nonfinite fold-normal source weather')
                start = 0
                while offset < end:
                    i, local = divmod(offset, season_days)
                    span = min(end - offset, season_days - local)
                    doy = self.first[i] + LOOKBACK + local
                    for f, v in enumerate(values):
                        self.prefix[i, f, doy:doy + span] = v[start:start + span]
                    start += span
                    offset += span
            if offset != len(dates):
                raise ValueError('Incomplete fold-normal weather calendar')
        for i, y in enumerate(years):
            days = np.arange(f'{y}-04-01', f'{y}-11-01', dtype='datetime64[D]')
            overlap = april_weather(y, days, donors, matrix, cache)
            prefix = self.prefix[i]
            first = self.first[i]
            for f, field in enumerate(ANOMALY_FIELDS):
                mapped = overlap[..., BASE_FIELDS.index(field)]
                if not np.allclose(mapped[LOOKBACK:], prefix[f, first + LOOKBACK:first + len(mapped)], rtol=1e-6, atol=1e-7):
                    raise ValueError(f'{y}: April warm-up differs from published {field}')
                prefix[f, first:first + LOOKBACK] = mapped[:LOOKBACK]
            np.cumsum(prefix, axis=1, out=prefix)
        self.samples = ((np.arange(367) >= self.first[:, None]) & (np.arange(367) <= self.last[:, None])).cumsum(axis=1)

    def apply(self, design, matrices, rows, reference, projection=None):
        """Replace anomaly columns for the selected years, preserving other inputs."""
        total = self.prefix[reference[0]].copy()
        for i in reference[1:]:
            total += self.prefix[i]
        counts = self.samples[reference].sum(axis=0)
        for i, out in zip(rows, matrices, strict=True):
            lo, hi = self.lower[i], self.upper[i]
            count = counts[hi] - counts[lo]
            own = i in reference
            if own:
                count -= self.samples[i, hi] - self.samples[i, lo]
            if (count <= 0).any():
                raise ValueError('Empty fold-local climate normal')
            days = len(lo)
            for f, (column, current) in enumerate(zip(self.columns, self.current_columns, strict=True)):
                amount = total[f, hi] - total[f, lo]
                if own:
                    amount -= self.prefix[i, f, hi] - self.prefix[i, f, lo]
                anomaly = design[i, :, current].reshape(days, self.cells) - amount / count[:, None]
                out[:, column if projection is None else projection[column]] = anomaly.ravel().astype(np.float32)
