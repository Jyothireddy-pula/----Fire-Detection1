"""
Export Module
Handles CSV and PDF export functionality
"""

import pandas as pd
from io import BytesIO
from typing import Dict, List
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer


class ExportService:
    """Service for exporting data to CSV and PDF formats"""
    
    def export_to_csv(self, data: List[Dict], filename: str = None) -> str:
        """
        Export data to CSV format
        
        Args:
            data: List of dictionaries to export
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"wildfire_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        return filename
    
    def export_predictions_to_csv(self, predictions: List[Dict], filename: str = None) -> str:
        """
        Export predictions to CSV with formatting
        
        Args:
            predictions: List of prediction dictionaries
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Flatten nested dictionaries
        flat_data = []
        for pred in predictions:
            flat_pred = {
                'timestamp': pred.get('timestamp'),
                'location': pred.get('location'),
                'latitude': pred.get('latitude'),
                'longitude': pred.get('longitude'),
                'risk_score': pred.get('risk_score'),
                'risk_level': pred.get('risk_level'),
                'confidence': pred.get('confidence'),
                'action': pred.get('action')
            }
            
            # Add weather data
            weather = pred.get('weather', {})
            for key, value in weather.items():
                flat_pred[f'weather_{key}'] = value
            
            # Add FWI components
            fwi = pred.get('fwi_components', {})
            for key, value in fwi.items():
                flat_pred[f'fwi_{key}'] = value
            
            flat_data.append(flat_pred)
        
        df = pd.DataFrame(flat_data)
        df.to_csv(filename, index=False)
        
        return filename
    
    def export_regional_scan_to_csv(self, scan_results: List[Dict], filename: str = None) -> str:
        """
        Export regional scan results to CSV
        
        Args:
            scan_results: List of regional scan results
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"regional_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        flat_data = []
        for result in scan_results:
            flat_result = {
                'region_name': result.get('region_name'),
                'latitude': result.get('latitude'),
                'longitude': result.get('longitude'),
                'risk_score': result.get('decision', {}).get('risk_score'),
                'risk_level': result.get('decision', {}).get('risk_level'),
                'confidence': result.get('decision', {}).get('confidence'),
                'temperature': result.get('weather', {}).get('temperature'),
                'humidity': result.get('weather', {}).get('humidity'),
                'wind_speed': result.get('weather', {}).get('wind_speed'),
                'rainfall': result.get('weather', {}).get('rainfall')
            }
            flat_data.append(flat_result)
        
        df = pd.DataFrame(flat_data)
        df.to_csv(filename, index=False)
        
        return filename
    
    def export_to_pdf(self, data: Dict, title: str = "Wildfire Risk Report", 
                     filename: str = None) -> str:
        """
        Export data to PDF format
        
        Args:
            data: Dictionary with report data
            title: Report title
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"wildfire_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = styles['Heading1']
        title_para = Paragraph(title, title_style)
        story.append(title_para)
        story.append(Spacer(1, 12))
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_para = Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal'])
        story.append(timestamp_para)
        story.append(Spacer(1, 12))
        
        # Create table from data
        table_data = [['Key', 'Value']]
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                table_data.append([str(key), str(value)])
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    table_data.append([f"{key}.{subkey}", str(subvalue)])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        doc.build(story)
        
        return filename
    
    def export_prediction_report_to_pdf(self, prediction: Dict, filename: str = None) -> str:
        """
        Export prediction report to PDF
        
        Args:
            prediction: Prediction dictionary
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Organize data for PDF
        report_data = {
            'Location': prediction.get('location', 'Unknown'),
            'Risk Score': f"{prediction.get('risk_score', 0):.2f}",
            'Risk Level': prediction.get('risk_level', 'Unknown'),
            'Confidence': f"{prediction.get('confidence', 0):.1%}",
            'Action': prediction.get('action', 'Unknown'),
            'Temperature': f"{prediction.get('weather', {}).get('temperature', 0):.1f}°C",
            'Humidity': f"{prediction.get('weather', {}).get('humidity', 0):.1f}%",
            'Wind Speed': f"{prediction.get('weather', {}).get('wind_speed', 0):.1f} km/h",
            'Rainfall': f"{prediction.get('weather', {}).get('rainfall', 0):.1f} mm",
            'FWI': f"{prediction.get('fwi_components', {}).get('FWI', 0):.1f}"
        }
        
        return self.export_to_pdf(report_data, "Wildfire Risk Prediction Report", filename)
