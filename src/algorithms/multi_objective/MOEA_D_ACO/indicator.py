def indicator_function(rule : list, next_term : tuple) -> int:
    """
    Returns 1 if the given term is part of the rule, otherwise returns 0.

    Args:
        rule (list): List of terms in the rule, where each term is a tuple (attribute, value).
        term (int): The term to check, represented as a tuple (attribute, value).
    Returns:
        int: 1 if the term is in the rule, 0 otherwise.
    """
    # TODO : change condition if class tuple is included in the rule
    if len(rule) < 2:
        return 0

    for term in rule:
        if (term[0] == next_term[0] and term[1] == next_term[1]):
            return 1
    return 0