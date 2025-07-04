import pandas as pd

def predict_function(X, archive, labels, archive_type='rules', prediction_strat='all'):
    """
    Predicts the class labels for the input data X using the rules in the archive.

    Args:
        X (pd.DataFrame): Input data for prediction.
        archive (list): Archive containing rules or rulesets.
        positive_class (str): The positive class label to consider.
        archive_type (str): Type of archive structure ('rules' or 'rulesets').
        prediction_strat (str): Strategy for prediction ('all', 'best', 'reference', 'voting').

    Returns:
        pd.Series: Predicted class labels.
    """

    if len(archive) == 0:
            raise ValueError("No rules found in the archive. Run the algorithm first.")
    
    y_preds = []

    # Order archive by f1 score if it contains rules
    archive = sorted(archive, key=lambda ant: ant['f1_score'], reverse=True)

    if archive_type == 'rules':
        if prediction_strat == 'all':
            # Predict using all rules in the archive    
             
            for _, row in X.iterrows():
                predicted_class = None
                for ant in archive:
                    if all(row[term[0]] == term[1] for term in ant['rule'][:-1]):
                        predicted_class = labels[0]
                        break
                if predicted_class is None:
                    predicted_class = labels[1]
    
                y_preds.append(predicted_class)

        elif prediction_strat == 'best':
             # Predict using the rule with the best f1 score

            for _, row in X.iterrows():
                predicted_class = None
                best_ant = archive[0]
                for ant in archive:
                    if ant['f1_score'] > best_ant['f1_score']:
                        best_ant = ant

                if all(row[term[0]] == term[1] for term in best_ant['rule'][:-1]):
                    predicted_class = labels[0]
                else:
                    predicted_class = labels[1]

                y_preds.append(predicted_class)
        
        elif prediction_strat == 'reference':
            # Predict using the closest rule to the ideal point
            for _, row in X.iterrows():
                predicted_class = None
                closest_ant = archive[0]
                min_distance = sum((f - 1) ** 2 for f in closest_ant['fitness']) ** 0.5

                for ant in archive:
                    distance = sum((f - 1) ** 2 for f in ant['fitness']) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_ant = ant

                if all(row[term[0]] == term[1] for term in closest_ant['rule'][:-1]):
                    predicted_class = labels[0]
                else:
                    predicted_class = labels[1]

                y_preds.append(predicted_class)

    elif archive_type == 'rulesets':
        if prediction_strat == 'all':
            # Predict using all rulesets in the archive

            for _, row in X.iterrows():
                predicted_class = None
                for ruleset in archive:
                    for ant in ruleset['ruleset']:
                        if all(row[term[0]] == term[1] for term in ant['rule'][:-1]):
                            predicted_class = labels[0]
                            break
                    if predicted_class is not None:
                        break
                if predicted_class is None:
                    predicted_class = labels[1]
    
                y_preds.append(predicted_class)

        elif prediction_strat == 'best':
            # Predict using the best ruleset based on f1 score

            best_ruleset = max(archive, key=lambda rs: rs['f1_score'])
            best_ruleset['ruleset'] = sorted(best_ruleset['ruleset'], key=lambda ant: ant['f1_score'], reverse=True)

            for _, row in X.iterrows():
                predicted_class = None
                
                for ant in best_ruleset['ruleset']:
                    if all(row[term[0]] == term[1] for term in ant['rule'][:-1]):
                        predicted_class = labels[0]
                        break

                if predicted_class is None:
                    predicted_class = labels[1]

                y_preds.append(predicted_class)

        elif prediction_strat == 'reference':
            # Predict using the closest ruleset to the ideal point

            for _, row in X.iterrows():
                predicted_class = None
                closest_ruleset = min(archive, key=lambda rs: sum(sum((f - 1) ** 2 for f in ant['fitness']) ** 0.5 for ant in rs['ruleset']) / len(rs['ruleset']))
                closest_ruleset['ruleset'] = sorted(closest_ruleset['ruleset'], key=lambda ant: ant['f1_score'], reverse=True)

                for ant in closest_ruleset['ruleset']:
                    if all(row[term[0]] == term[1] for term in ant['rule'][:-1]):
                        predicted_class = labels[0]
                        break

                if predicted_class is None:
                    predicted_class = labels[1]

                y_preds.append(predicted_class)

    return pd.Series(y_preds, index=X.index)


