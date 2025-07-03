from shiny import App, ui, render, reactive

# Sample matrix data
matrix_data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

app_ui = ui.page_fluid(
    ui.h2("Clickable Matrix with Highlight"),
    # Define CSS for the highlighted cell
    ui.tags.head(
        ui.tags.style("""
            table {
                border-collapse: collapse;
                width: 100%;
            }
            td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
                cursor: pointer; /* Indicate clickable */
                transition: background-color 0.2s ease; /* Smooth transition for highlight */
            }
            .highlighted-cell {
                background-color: yellow;
            }
        """)
    ),
    ui.output_ui("matrix_output"),
    # JavaScript to handle clicks and highlighting
    ui.tags.script("""
        // Store the previously highlighted cell element
        let previouslyHighlightedCell = null;

        // Function to handle cell clicks
        function handleCellClick(event) {
            const clickedCell = event.currentTarget; // The td element

            // If there was a previously highlighted cell, remove its class
            if (previouslyHighlightedCell) {
                previouslyHighlightedCell.classList.remove('highlighted-cell');
            }

            // Add the highlight class to the currently clicked cell
            clickedCell.classList.add('highlighted-cell');

            // Update the previously highlighted cell for the next click
            previouslyHighlightedCell = clickedCell;

            // Optional: Send the cell ID to the Shiny server if needed for other server-side logic
            const cellId = clickedCell.dataset.cellId; // Get the data-cell-id attribute
            if (cellId) {
                Shiny.setinput('clicked_cell_js', cellId);
            }
        }

        // Wait for the DOM to be fully loaded and Shiny to be ready
        document.addEventListener('DOMContentLoaded', function() {
            // Get the table element (or its container)
            const matrixTable = document.getElementById('matrix_table');

            if (matrixTable) {
                // Add click listener to the table and use event delegation
                matrixTable.addEventListener('click', function(event) {
                    // Check if the clicked element is a td
                    if (event.target.tagName === 'TD') {
                        handleCellClick(event);
                    }
                });

                // Optional: Highlight the first cell initially on page load
                // Find the first cell (td with data-cell-id="0-0")
                const firstCell = matrixTable.querySelector('td[data-cell-id="0-0"]');
                if (firstCell) {
                    firstCell.classList.add('highlighted-cell');
                    previouslyHighlightedCell = firstCell;
                    Shiny.setinput('clicked_cell_js', '0-0'); // Inform server about initial highlight
                }
            }
        });
    """)
)

def server(input, output, session):
    # This reactive value will be updated by the JavaScript
    clicked_cell_from_js = reactive.Value("0-0") # Initialize with default for consistency

    # A reactive effect to demonstrate receiving the clicked cell ID from JavaScript
    @reactive.Effect
    @reactive.event(input.clicked_cell_js)
    def _():
        print(f"Cell clicked (from JS): {input.clicked_cell_js()}")
        clicked_cell_from_js.set(input.clicked_cell_js())
        # You could add further server-side logic here based on the clicked cell
        ui.notification_show(f"Cell {input.clicked_cell_js()} was clicked!", type="message", duration=1.5)


    @render.ui
    def matrix_output():
        table_rows = []
        for r_idx, row in enumerate(matrix_data):
            row_cells = []
            for c_idx, cell_value in enumerate(row):
                cell_id = f"{r_idx}-{c_idx}"
                row_cells.append(
                    ui.tags.td(
                        str(cell_value),
                        # Add a data attribute to identify the cell in JavaScript
                        data_cell_id=cell_id
                    )
                )
            table_rows.append(ui.tags.tr(*row_cells))

        return ui.tags.table(
            ui.tags.tbody(*table_rows),
            id="matrix_table" # Assign an ID to the table for JavaScript to find it
        )

app = App(app_ui, server)