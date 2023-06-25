import os
import pandas as pd

# Set the working directory to the directory of the script file.
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def filterDialogue(input_file, output_file, allowedTenses):
    """
    Filter a Harry Potter dialogue file to include only lines that contain
    substrings that are surrounded by asteriks and then are surrounded by
    opening (and closing) parenthesis.

    Args:
        input_file (str): Path to the input Excel file.
        output_file (str): Path to the output Excel file.

    Returns:
        None
    """

    # Read the input file into a pandas dataframe.
    df = pd.read_excel(input_file, header=0)

    # Remove tenses that are not in the allowed list.
    df = df[df["source tense"].isin(allowedTenses)]

    """
    Check if there are an uneven amount of parentheses before an asterisk in the given string.

    :param myString: A string to search for parentheses and asterisks.
    :type myString: str
    :return: True if there are parentheses before an asterisk, False otherwise.
    :rtype: bool
    """

    def existParaBeforeAsterix(myString):
        myString = list(myString)
        parFound = 0
        for i in range(len(myString)):
            myChar = myString[i]
            if myChar == '"':
                parFound += 1
            if myChar == "*":
                if parFound % 2 == 0:
                    return False
                else:
                    return True
        return False

    # Filter out rows where there are no uneven amount of parentheses before an asterisk in the given string.
    df = df[df["full fragment"].apply(existParaBeforeAsterix)]
    print(df)

    # Write the filtered dataframe to an Excel file.
    df.to_excel(output_file, index=False)

    return df


def renameColumn(df, tag):
    """
    Renames all columns in the given DataFrame by prefixing them with the given tag,
    unless the column name contains the string "source". Returns the updated DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to rename columns for.
        tag (str): The prefix to add to each column name.

    Returns:
        pandas.DataFrame: The updated DataFrame with renamed columns.
    """

    originalColumns = df.columns
    new_columns = [
        tag + "_" + col if "source" not in col else col for col in originalColumns
    ]

    df.columns = new_columns

    return df


def mergeColumns(dfs, output_file):
    """
    Merge the given dataframes and write the merged dataframe to an Excel file.

    Args:
        dfs (list): A list of pandas dataframes to merge.
        output_file (str): The filepath to write the merged dataframe to.

    Returns:
        pandas.DataFrame: The merged dataframe.
    """

    # Iterate over the remaining dataframes and merge them with the merged dataframe
    commonColumns = dfs[0].columns
    for df in dfs[1:]:
        commonColumns = commonColumns.intersection(df.columns)

    dfMerged = dfs[0]
    for df in dfs[1:]:
        dfMerged = pd.merge(dfMerged, df, on=list(commonColumns), how="outer")

    # Reorder the columns and sort df based on source id
    desiredColumns = commonColumns
    for df in dfs:
        desiredColumns = desiredColumns.union(df.columns.difference(commonColumns))
    dfMerged = dfMerged.reindex(columns=desiredColumns)
    dfMerged = dfMerged.sort_values("source id")

    #
    print(dfMerged)

    # Write the filtered dataframe to an Excel file.
    dfMerged.to_excel(output_file, index=False)

    return dfMerged


allowedTenses = [
    "simple present",
    "simple past",
    "present perfect",
    "present continuous",
    "past perfect",
    "past continuous",
    "present perfect continuous",
]

dfNL = filterDialogue(
    "res/Harry Potter-all-nl.xlsx",
    "dist/Harry Potter-dialogue-nl.xlsx",
    allowedTenses,
)
dfDE = filterDialogue(
    "res/Harry Potter-all-de.xlsx",
    "dist/Harry Potter-dialogue-de.xlsx",
    allowedTenses,
)

dfNL = renameColumn(dfNL, "nl")
dfDE = renameColumn(dfDE, "de")

dfEN = mergeColumns([dfNL, dfDE], "dist/Harry Potter-dialogue-en-nl-de.xlsx")
