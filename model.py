import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set the working directory to the directory of the script file.
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def data_vis(data):
    def correlate(data):
        plt.clf()  # Clear the current figure
        plt.figure(figsize=(60, 45))

        # Encode and get the correlation matrix
        df_encoded = pd.get_dummies(data)
        correlation_matrix = df_encoded.corr()

        # Visualize the correlation heatmap
        ax = sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            square=True,
            fmt=".2f",
            annot=True,
        )
        ax.invert_yaxis()
        ax.invert_xaxis()

        # Highlight the interesting points
        rect = plt.Rectangle(
            (27, 1),
            13,
            24,
            fill=False,
            edgecolor="red",
            lw=2,
        )
        ax.add_patch(rect)

        plt.tight_layout()  # Adjust subplot parameters to fit the figure area
        plt.savefig("doc/images/correlation_heatmap.png")

    def create_distribution_chart(column_name, dataframe):
        plt.clf()  # Clear the current figure
        plt.figure(figsize=(12, 9))

        # Select the specified column from the DataFrame
        column_data = dataframe[column_name]

        # Calculate the count for each unique value in the column
        value_counts = column_data.value_counts().sort_values(ascending=False)

        # Create the bar plot with descending sorting by count
        sns.barplot(
            x=value_counts,
            y=value_counts.index,
            color="#607c8e",
            order=value_counts.index,  # Sort bars by count
        )

        # Set chart labels
        plt.xlabel("Count")

        # Save the chart
        plt.subplots_adjust(left=0.35)
        plt.savefig("doc/images/dist_" + column_name + ".png")

        # Print the distribution
        print("Distribution for column:", column_name)
        print(value_counts)

    correlate(data)
    create_distribution_chart("DAA", data)
    create_distribution_chart("source tense", data)
    create_distribution_chart("nl_tense", data)
    create_distribution_chart("de_tense", data)


def rfModel(df, trainAttr, targetAttr, k=5):
    def initData(df, trainAttr):
        # Define the column transformer for one-hot encoding
        categorical_features = [
            i
            for i, col in enumerate(trainAttr)
            if col in df.columns and df[col].dtype == "object"
        ]
        preprocessor = ColumnTransformer(
            [("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
            remainder="passthrough",
        )

        return preprocessor

    def kFoldRF(preprocessor, df, trainAttr, targetAttr, k=5, iter=10):
        # Extract the selected columns and the target variable
        X = df[trainAttr].values
        y = df[targetAttr].values

        # Setup score lists
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Perform K-fold cross-validation iter times
        for _ in range(iter):
            # Perform one K-fold cross-validation
            rf = RandomForestClassifier()
            kf = KFold(n_splits=k, shuffle=True)

            # Setup score lists
            local_accuracy_scores = []
            local_precision_scores = []
            local_recall_scores = []
            local_f1_scores = []
            for train_index, test_index in kf.split(X):
                # Split the data
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Preprocess the data (perform one-hot encoding)
                X_train = preprocessor.fit_transform(X_train)
                X_test = preprocessor.transform(X_test)

                # Fit the Random Forest model on the training data and make predictions on the test data
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                # Calculate the scores
                accuracy = accuracy_score(y_test, y_pred)
                local_accuracy_scores.append(accuracy)
                precision = precision_score(
                    y_test, y_pred, average="macro", zero_division=1
                )
                local_precision_scores.append(precision)
                recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
                local_recall_scores.append(recall)
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=1)
                local_f1_scores.append(f1)

            accuracy_scores.append(
                sum(local_accuracy_scores) / len(local_accuracy_scores)
            )
            precision_scores.append(
                sum(local_precision_scores) / len(local_precision_scores)
            )
            recall_scores.append(sum(local_recall_scores) / len(local_recall_scores))
            f1_scores.append(sum(local_f1_scores) / len(local_f1_scores))

        return rf, accuracy_scores, precision_scores, recall_scores, f1_scores

    def createCPM(rf, preprocessor, df, trainAttr, targetAttr):
        # Generate all possible combinations of inputs
        all_combinations = list(product(*preprocessor.transformers_[0][1].categories_))
        results_df = pd.DataFrame(all_combinations, columns=trainAttr)

        # Add the target probabilities for each combination
        target_probabilities = rf.predict_proba(preprocessor.transform(results_df))
        for i, target in enumerate(rf.classes_):
            results_df[f"probability_{target}"] = target_probabilities[:, i]

        # Visualize the probabilities
        if len(trainAttr) <= 2:
            plt.clf()  # Clear the current figure
            plt.figure(figsize=(18, 1.5 * len(results_df) / len(trainAttr)))

            # Plot the probabilities
            probability_columns = [f"probability_{target}" for target in rf.classes_]
            probabilities = results_df[probability_columns].values.reshape(
                len(all_combinations), -1
            )

            # Visualize the probabilities using a heatmap
            sns.heatmap(
                probabilities,
                xticklabels=rf.classes_,
                yticklabels=results_df[trainAttr].astype(str).agg(" + ".join, axis=1),
                cmap="rocket_r",
                annot=True,
                fmt=".2f",
            )
            plt.xticks(rotation=90)
            plt.xlabel("Target Value")
            plt.ylabel("Feature Value(s)")
            plt.tight_layout()  # Adjust subplot parameters to fit the figure area
            plt.savefig(
                "doc/images/pTable "
                + "+".join(trainAttr)
                + " to "
                + targetAttr
                + ".png"
            )

        return results_df

    # Get the data
    df = df.dropna(subset=[targetAttr])
    # df = df[df["source tense"].isin(["present perfect", "past perfect"])]  # Use only present perfect or past perfect
    # df = df[df["DAA - Extra"] != ""] # Use only sentences with clauses

    # Define the column transformer
    preprocessor = initData(df, trainAttr)
    rf, accuracy_scores, precision_scores, recall_scores, f1_scores = kFoldRF(
        preprocessor, df, trainAttr, targetAttr, k
    )
    results_df = createCPM(rf, preprocessor, df, trainAttr, targetAttr)

    # Print the results
    # print("Average accuracy:", average_accuracy)
    # print("Average precision:", average_precision)
    # print("Average recall:", average_recall)
    # print("Average f1:", average_f1)
    # print()
    # results_df.to_csv(
    #     "pTable " + "+".join(trainAttr) + " to " + targetAttr + ".tsv",
    #     sep="\t",
    #     index=False,
    # )
    return (accuracy_scores, precision_scores, recall_scores, f1_scores)


def run_models(data):
    def vis_scores(
        settings,
        accuracy_scores,
        precision_scores,
        recall_scores,
        f1_scores,
    ):
        plt.clf()  # Clear the current figure
        fig, axs = plt.subplots(1, 4, figsize=(18, 9), sharex=True)

        num_models = len(settings)
        indices = np.arange(num_models)
        bar_width = 1
        colors = plt.cm.get_cmap("Pastel1").colors

        # Plot accuracy scores
        axs[0].bar(indices, accuracy_scores, bar_width, color=colors)
        axs[0].set_ylabel("Accuracy")
        axs[0].set_ylim(0, 1)

        # Plot precision scores
        axs[1].bar(indices, precision_scores, bar_width, color=colors)
        axs[1].set_ylabel("Precision")
        axs[1].set_ylim(0, 1)

        # Plot recall scores
        axs[2].bar(indices, recall_scores, bar_width, color=colors)
        axs[2].set_ylabel("Recall")
        axs[2].set_ylim(0, 1)

        # Plot F1 scores
        axs[3].bar(indices, f1_scores, bar_width, color=colors)
        axs[3].set_ylabel("F1 Score")
        axs[3].set_ylim(0, 1)

        # Remove x-axis labels
        plt.setp(axs, xticks=[])

        # Create labels for legend
        x_labels = [
            ", ".join(str(label) for label in setting[0]) for setting in settings
        ]

        # Add legend with labeled colors
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color) for color in colors[:num_models]
        ]
        labels = x_labels
        fig.legend(handles, labels, loc="upper right")

        # Save the chart
        plt.tight_layout()  # Adjust subplot parameters to fit the figure area
        plt.savefig("doc/images/model_scores_" + settings[0][1] + ".png")

        return

    def run_anova(settings, accuracy_scores, f1_scores):
        print(accuracy_scores)
        print(f1_scores)

        print(settings)

        # Reshape the data for ANOVA
        accuracy_data = (
            accuracy_scores[0]
            + accuracy_scores[1]
            + accuracy_scores[2]
            + accuracy_scores[3]
        )
        f1_data = f1_scores[0] + f1_scores[1] + f1_scores[2] + f1_scores[3]
        group_labels = (
            ["Group 1"] * len(accuracy_scores[0])
            + ["Group 2"] * len(accuracy_scores[1])
            + ["Group 3"] * len(accuracy_scores[2])
            + ["Group 4"] * len(accuracy_scores[3])
        )

        # Create a DataFrame with the data
        df = pd.DataFrame(
            {"Accuracy": accuracy_data, "F1": f1_data, "Group": group_labels}
        )

        # Fit the ANOVA model
        model = ols("Accuracy ~ C(Group)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Print the ANOVA table
        print(anova_table)

        # Fit the ANOVA model for F1 scores
        model_f1 = ols("F1 ~ C(Group)", data=df).fit()
        anova_table_f1 = sm.stats.anova_lm(model_f1, typ=2)

        # Print the ANOVA table for F1 scores
        print(anova_table_f1)

    for target in ["nl_tense", "de_tense"]:
        # Define empty lists
        settings = []
        average_accuracy_scores = []
        average_precision_scores = []
        average_recall_scores = []
        average_f1_scores = []

        inter_model_accuracy_scores = []
        inter_model_precision_scores = []
        inter_model_recall_scores = []
        inter_model_f1_scores = []

        # Run the model
        for features in [
            ["source tense"],
            ["source tense", "DAA - Simple"],
            ["source tense", "DAA"],
            ["source tense", "DAA", "DAA - Extra"],
        ]:
            accuracy_scores, precision_scores, recall_scores, f1_scores = rfModel(
                data,
                features,
                target,
            )

            settings.append([features, target])

            average_accuracy_scores.append(sum(accuracy_scores) / len(accuracy_scores))
            average_precision_scores.append(
                sum(precision_scores) / len(precision_scores)
            )
            average_recall_scores.append(sum(recall_scores) / len(recall_scores))
            average_f1_scores.append(sum(f1_scores) / len(f1_scores))

            inter_model_accuracy_scores.append(accuracy_scores)
            inter_model_precision_scores.append(precision_scores)
            inter_model_recall_scores.append(recall_scores)
            inter_model_f1_scores.append(f1_scores)

        # Visualize the results
        vis_scores(
            settings,
            average_accuracy_scores,
            average_precision_scores,
            average_recall_scores,
            average_f1_scores,
        )
        run_anova(
            settings,
            inter_model_accuracy_scores,
            inter_model_f1_scores,
        )


# Set the new font size for the plot
plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 2})

data = pd.read_excel("dist/HP_DAA_simple.xlsx")
data_vis(data)
run_models(data)
