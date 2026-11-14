"""Dataset names as they appear in the paper's text, tables and figures."""

SHORT = {"airquality": "airqual", "blogfeedback": "blogfeed", "cpusmallscale": "cpusmall"}


def display_name(problem):
    """Paper name of a problem key, e.g. CTSLICES_GROUPED -> ctslices."""
    key = problem.replace("_GROUPED", "").lower()
    return SHORT.get(key, key)
