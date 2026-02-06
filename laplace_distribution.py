import math
def calculate_risk_metrics(alpha, scale, mean, pv):
    """
    Calculates VaR and CVaR metrics based on the provided source logic.
    """
    try:
        # Calculate VaR Rate (Log Return threshold)
        var_rate = math.log((1 - alpha) * 2) * scale + mean
        results = {
            "var_rate_raw": var_rate,
            "is_risk": False,
            "metrics": {}
        }
        # The source logic checks if the calculated rate implies a loss (negative return)
        if var_rate < 0:
            results["is_risk"] = True
            # Calculate VaR Monetary Value
            var_pt = pv * (math.exp(var_rate) - 1)
            # Calculate CVaR Rate
            cvar_rate_val = var_rate * (-1) + scale 
            # Calculate CVaR Monetary Value
            cvar_pt = pv * (math.exp(-cvar_rate_val) - 1)
            # Map to display values (reversing signs where source code did)
            results["metrics"] = {
                "VaR Rate": var_rate * -1,      # Displayed as positive magnitude
                "VaR Reserved": var_pt * -1,    # Displayed as positive magnitude
                "CVaR Rate": cvar_rate_val,     # Displayed as is
                "Mean Big Loss": cvar_pt * -1   # Displayed as positive magnitude
            }
        return results
    except ValueError:
        # Handle math domain errors (e.g. log of negative number)
        return None