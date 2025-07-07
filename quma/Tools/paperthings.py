def array_to_latex_table(array,headers=None):
    """
    Converts a 2D array (list of lists) into a LaTeX table.
    If the first column values contain the prefix "alpha_", it is removed.
    A header row is generated dynamically based on the number of columns.
    
    Parameters:
        array (list of lists): 2D array to convert into a LaTeX table.
        
    Returns:
        None: Prints the LaTeX table code.
    """
    # Check if the array is empty or has no columns
   # if not array or not array[0]:
    #    print("Empty or invalid array provided.")
     #   return
    
    n_cols = len(array[0])
    
    # Build the header row dynamically: first column header remains blank.
    if headers is None:
        headers = [""] + [f"Col{j+1}" for j in range(1, n_cols)]
    else:
        headers = [""] + headers
    header_line = " & ".join(headers) + " \\\\\n"
    
    # Start building the LaTeX table string
    latex_str = "\\begin{table}\n"
    latex_str += "    \\centering\n"
    #latex_str += "renewcommand{\arraystretch}{1.2}"
    latex_str += "    \\begin{tabular}{" + "c" * n_cols + "}\n"
    latex_str += "        \\hline\n"
    # Add the dynamically created header row
    latex_str += "        " + header_line
    latex_str += "        \\hline\n"
    
    # Process each row in the array
    for row in array:
        # Process the first element: remove "alpha_" if present.
        first_item = row[0]
        if isinstance(first_item, str) and first_item.startswith("alpha_"):
            first_item = first_item.replace("alpha_", "")
        # Convert each item in the row to a string (first element processed separately)
        row_items = [str(first_item)] + [str(item) for item in row[1:]]
        # Join them with " & " and append LaTeX newline
        row_line = " & ".join(row_items) + " \\\\\n"
        latex_str += "        " + row_line
    
    # Finish table with a bottom horizontal line and caption/label
    latex_str += "        \\hline\n"
    latex_str += "    \\end{tabular}\n"
    latex_str += "    \\caption{Generated table with dynamic headers}\n"
    latex_str += "    \\label{tab:generated}\n"
    latex_str += "\\end{table}"
    
    print(latex_str)