# CSV Editor Usage

`CsvEditor` supports three operations:

1. `append_row(row)` appends a row to an existing CSV file.
2. `update_cell(row_index, column_index, value)` updates a specific cell.
3. `read_rows()` loads CSV content into nested string lists.

This matches the internal lightweight workflow for editing tabular test fixtures.
