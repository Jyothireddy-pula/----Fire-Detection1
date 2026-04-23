"""
Explainability Engine Module
Generates explanations for model predictions using rule-based analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional


class ExplainabilityEngine:
    """
    Generates human-readable explanations for wildfire risk predictions.
    Works with Fuzzy, ANFIS, and PSO-ANFIS models.
    """

    def __init__(self, fuzzy_system, anfis_model=None):
        """
        Initialize explainability engine.

        Args:
            fuzzy_system: FuzzyWildfireSystem instance
            anfis_model: Optional ANFIS model instance
        """
        self.fuzzy_system = fuzzy_system
        self.anfis_model = anfis_model

    def explain_prediction(self, prediction_result: Dict, weather_data: Dict,
                          fwi_components: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            prediction_result: Result from predict_pipeline
            weather_data: Weather input dict
            fwi_components: FWI components dict

        Returns:
            Explanation dict with factor_importance, fired_rules, summary, etc.
        """
        # Compute factor importance
        factor_importance = self._compute_factor_importance(weather_data, fwi_components)

        # Explain fired fuzzy rules
        fuzzy_details = prediction_result.get('fuzzy_details', {})
        fired_rules = self._explain_fired_rules(fuzzy_details)

        # ANFIS layer explanation if available
        anfis_output = prediction_result.get('anfis_output')
        anfis_explanation = None
        if self.anfis_model and anfis_output is not None:
            anfis_explanation = self._explain_anfis_layers(weather_data, anfis_output)

        # Generate summary
        summary = self._generate_summary(factor_importance, fired_rules, prediction_result)

        # Compute confidence
        confidence = self._compute_confidence(prediction_result)

        return {
            'factor_importance': factor_importance,
            'fired_rules': fired_rules,
            'anfis_explanation': anfis_explanation,
            'summary': summary,
            'confidence': confidence,
            'model_used': prediction_result.get('model_used', 'fuzzy')
        }

    def _compute_factor_importance(self, weather_data: Dict,
                                    fwi_components: Dict) -> List[Dict]:
        """Compute which factors contribute most to risk."""
        factors = {}

        # Weather factors
        temp = weather_data.get('temperature', 25)
        humidity = weather_data.get('humidity', 50)
        wind = weather_data.get('wind_speed', 10)
        rain = weather_data.get('rainfall', 0)

        # FWI components
        ffmc = fwi_components.get('FFMC', 85)
        isi = fwi_components.get('ISI', 5)
        bui = fwi_components.get('BUI', 10)
        fwi = fwi_components.get('FWI', 5)

        # Temperature impact
        if temp >= 40:
            temp_impact = 0.95
        elif temp >= 35:
            temp_impact = 0.80
        elif temp >= 30:
            temp_impact = 0.50
        elif temp >= 25:
            temp_impact = 0.25
        else:
            temp_impact = 0.10
        factors['Temperature'] = {'value': temp, 'impact': temp_impact}

        # Humidity impact (inverse - low humidity = high risk)
        if humidity <= 20:
            hum_impact = 0.95
        elif humidity <= 30:
            hum_impact = 0.80
        elif humidity <= 50:
            hum_impact = 0.50
        elif humidity <= 70:
            hum_impact = 0.25
        else:
            hum_impact = 0.10
        factors['Humidity'] = {'value': humidity, 'impact': hum_impact}

        # Wind impact
        if wind >= 30:
            wind_impact = 0.90
        elif wind >= 20:
            wind_impact = 0.70
        elif wind >= 15:
            wind_impact = 0.45
        else:
            wind_impact = 0.15
        factors['Wind Speed'] = {'value': wind, 'impact': wind_impact}

        # Rainfall impact (inverse - no rain = high risk)
        if rain <= 0.1:
            rain_impact = 0.85
        elif rain <= 2:
            rain_impact = 0.50
        elif rain <= 5:
            rain_impact = 0.20
        else:
            rain_impact = 0.05
        factors['Rainfall'] = {'value': rain, 'impact': rain_impact}

        # FFMC impact (primary fire indicator)
        if ffmc >= 90:
            ffmc_impact = 0.98
        elif ffmc >= 85:
            ffmc_impact = 0.85
        elif ffmc >= 80:
            ffmc_impact = 0.65
        elif ffmc >= 70:
            ffmc_impact = 0.35
        else:
            ffmc_impact = 0.10
        factors['FFMC'] = {'value': ffmc, 'impact': ffmc_impact}

        # ISI impact (spread potential)
        if isi >= 15:
            isi_impact = 0.90
        elif isi >= 10:
            isi_impact = 0.70
        elif isi >= 5:
            isi_impact = 0.45
        else:
            isi_impact = 0.15
        factors['ISI'] = {'value': isi, 'impact': isi_impact}

        # BUI impact (fuel availability)
        if bui >= 40:
            bui_impact = 0.85
        elif bui >= 20:
            bui_impact = 0.60
        elif bui >= 10:
            bui_impact = 0.35
        else:
            bui_impact = 0.10
        factors['BUI'] = {'value': bui, 'impact': bui_impact}

        # FWI overall impact
        if fwi >= 25:
            fwi_impact = 1.0
        elif fwi >= 15:
            fwi_impact = 0.85
        elif fwi >= 10:
            fwi_impact = 0.65
        elif fwi >= 5:
            fwi_impact = 0.35
        else:
            fwi_impact = 0.10
        factors['FWI'] = {'value': fwi, 'impact': fwi_impact}

        # Sort by impact
        sorted_factors = sorted(factors.items(), key=lambda x: x[1]['impact'], reverse=True)

        return [
            {'name': k, 'value': v['value'], 'impact': v['impact']}
            for k, v in sorted_factors
        ]

    def _explain_fired_rules(self, fuzzy_details: Dict) -> List[Dict]:
        """Explain which fuzzy rules fired and their contribution."""
        fired_rules_raw = fuzzy_details.get('fired_rules', [])
        explanations = []

        for rule_info in fired_rules_raw[:6]:
            rule = rule_info.get('rule', {})
            strength = rule_info.get('firing_strength', 0)

            if strength > 0.15:
                conditions = rule.get('conditions', {})
                output = rule.get('output', 'unknown')
                reasoning = rule.get('reasoning', '')

                # Build readable conditions string
                cond_parts = []
                for var, term in conditions.items():
                    cond_parts.append(f"{var}={term}")

                explanations.append({
                    'conditions': ', '.join(cond_parts),
                    'output': output,
                    'strength': round(strength, 3),
                    'reasoning': reasoning
                })

        return explanations

    def _explain_anfis_layers(self, weather_data: Dict, anfis_output: float) -> Dict:
        """Explain ANFIS prediction using layer-by-layer analysis."""
        if not self.anfis_model or not self.anfis_model.is_trained:
            return None

        return {
            'output': anfis_output,
            'note': 'ANFIS model was used for this prediction. '
                    'The output is a weighted combination of rule consequents, '
                    'where weights are normalized firing strengths from the fuzzy layer.'
        }

    def _generate_summary(self, factor_importance: List[Dict],
                          fired_rules: List[Dict],
                          prediction_result: Dict) -> str:
        """Generate human-readable summary."""
        risk_level = prediction_result.get('linguistic_risk_level', 'Unknown')
        risk_score = prediction_result.get('risk_score', 0)

        top_factors = factor_importance[:3]
        top_factor = top_factors[0] if top_factors else None

        if risk_score >= 0.75:
            summary = f"HIGH RISK: {risk_level} conditions detected. "
            if top_factor:
                summary += f"Primary driver is {top_factor['name']} at {top_factor['value']:.1f} "
                summary += f"(impact: {top_factor['impact']:.0%}). "
            if len(top_factors) > 1:
                summary += f"Secondary factor {top_factors[1]['name']} (impact: {top_factors[1]['impact']:.0%}) also contributes. "
            if fired_rules:
                summary += f"Key fuzzy rule: {fired_rules[0]['reasoning'][:100]}..."
        elif risk_score >= 0.5:
            summary = f"MODERATE RISK: {risk_level} conditions. "
            if top_factor:
                summary += f"{top_factor['name']} at {top_factor['value']:.1f} is the main contributor "
                summary += f"with {top_factor['impact']:.0%} impact. "
            summary += "Conditions warrant monitoring but immediate danger is low."
        elif risk_score >= 0.25:
            summary = f"LOW RISK: {risk_level} conditions. "
            if top_factor:
                summary += f"{top_factor['name']} ({top_factor['impact']:.0%} impact) is the primary factor. "
            summary += "No immediate fire danger, maintain normal precautions."
        else:
            summary = f"MINIMAL RISK: {risk_level}. "
            summary += "All environmental factors are within safe ranges. No fire risk detected."

        return summary

    def _compute_confidence(self, prediction_result: Dict) -> float:
        """Compute confidence score for the prediction."""
        fuzzy_details = prediction_result.get('fuzzy_details', {})
        rule_firing = fuzzy_details.get('rule_firing_strength', 0)

        # Confidence based on rule firing strength and consistency
        confidence = min(0.95, 0.5 + 0.4 * rule_firing)

        return round(confidence, 3)

    def explain_why_high_risk(self, prediction_result: Dict,
                               explanation: Dict) -> List[str]:
        """Generate 'why' explanations for high-risk alerts."""
        factors = explanation.get('factor_importance', [])
        fired_rules = explanation.get('fired_rules', [])

        why_explanations = []

        # Factor-based explanation
        for factor in factors[:3]:
            if factor['impact'] > 0.6:
                if factor['name'] == 'Temperature':
                    why_explanations.append(
                        f"Temperature is elevated at {factor['value']:.1f}°C, "
                        f"contributing {factor['impact']:.0%} to fire risk"
                    )
                elif factor['name'] == 'Humidity':
                    why_explanations.append(
                        f"Low humidity at {factor['value']:.1f}% creates dry conditions, "
                        f"contributing {factor['impact']:.0%} to fire risk"
                    )
                elif factor['name'] == 'FFMC':
                    why_explanations.append(
                        f"FFMC at {factor['value']:.1f} indicates very dry fine fuels, "
                        f"contributing {factor['impact']:.0%} to fire risk"
                    )
                elif factor['name'] == 'Wind Speed':
                    why_explanations.append(
                        f"High wind speed at {factor['value']:.1f} km/h increases spread potential, "
                        f"contributing {factor['impact']:.0%} to fire risk"
                    )
                elif factor['name'] == 'Rainfall':
                    why_explanations.append(
                        f"Absence of rainfall keeps fuels dry, "
                        f"contributing {factor['impact']:.0%} to fire risk"
                    )

        # Rule-based explanation
        if fired_rules:
            top_rule = fired_rules[0]
            why_explanations.append(
                f"Primary fuzzy rule triggered: {top_rule['reasoning']}"
            )

        return why_explanations

    def explain_scenario_delta(self, base_risk: float, scenario_risk: float,
                               base_explanation: Dict,
                               scenario_explanation: Dict) -> str:
        """Explain why a scenario differs from baseline."""
        delta = scenario_risk - base_risk

        if abs(delta) < 0.05:
            return "This scenario has similar risk to baseline conditions."

        direction = "increases" if delta > 0 else "decreases"
        magnitude = "significantly" if abs(delta) > 0.3 else "moderately"

        base_top = base_explanation.get('factor_importance', [{}])[0].get('name', 'unknown')
        scenario_top = scenario_explanation.get('factor_importance', [{}])[0].get('name', 'unknown')

        if base_top != scenario_top:
            return (f"This scenario {magnitude} {direction} risk (+{delta:.2f}) because "
                    f"{scenario_top} becomes the primary driver instead of {base_top}.")
        else:
            return (f"This scenario {magnitude} {direction} risk (+{delta:.2f}). "
                    f"{scenario_top} impact has changed compared to baseline.")
