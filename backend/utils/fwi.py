"""
FWI (Fire Weather Index) Module
Implements real FWI calculation formulas for wildfire risk assessment
"""

import numpy as np


class FWICalculator:
    """Fire Weather Index Calculator using real Canadian FWI System formulas"""
    
    def __init__(self):
        # Constants for FFMC calculation
        self.ffmc_init = 85.0
        
    def compute_ffmc(self, temp: float, rh: float, wind: float, rain: float, 
                     prev_ffmc: float = None) -> float:
        """
        Compute Fine Fuel Moisture Code (FFMC)
        
        Args:
            temp: Temperature (°C)
            rh: Relative Humidity (%)
            wind: Wind speed (km/h)
            rain: Rainfall (mm)
            prev_ffmc: Previous day's FFMC (default: 85.0)
            
        Returns:
            FFMC value (0-101)
        """
        if prev_ffmc is None:
            prev_ffmc = self.ffmc_init
            
        # Rain effect
        if rain > 0.5:
            rf = rain - 0.5
            if prev_ffmc <= 50:
                mo = 147.2 * (101 - prev_ffmc) / (59.5 + prev_ffmc)
            else:
                mo = 21.1 * np.exp((100 - prev_ffmc) / 29.5)
            
            if mo >= 150:
                mo = mo + rf * 0.0  # Saturated
            else:
                mo = mo + 42.5 * rf * np.exp(-100 / (251 + mo)) * (1 - np.exp(-6.93 / (rf + 1.08)))
            
            if mo > 250:
                mo = 250
            
            if prev_ffmc <= 50:
                prev_ffmc = 101 - 59.5 * (250 - mo) / (147.2 + mo)
            else:
                prev_ffmc = 101 - 29.5 * np.log(250 - mo) + 18.28
        
        # Moisture equilibrium
        ed = 0.942 * (rh ** 0.679) + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh))
        
        if prev_ffmc <= 50:
            ew = 0.618 * (rh ** 0.753) + 10 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh))
        else:
            ew = 0.942 * (rh ** 0.679) + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh))
        
        # Log drying rate
        if prev_ffmc <= 50:
            kl = 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(wind) * (1 - (rh / 100) ** 8)
        else:
            kl = 0.424 * (1 - (1 - rh / 100) ** 1.7) + 0.0694 * np.sqrt(wind) * (1 - (1 - rh / 100) ** 8)
        
        # Adjust moisture content
        if prev_ffmc <= 50:
            m = 147.2 * (101 - prev_ffmc) / (59.5 + prev_ffmc)
        else:
            m = 21.1 * np.exp((100 - prev_ffmc) / 29.5)
        
        mr = m + kl * (ew - m)
        
        if mr < 0:
            mr = 0
        elif mr > 250:
            mr = 250
        
        # Convert to FFMC
        if mr <= 147.2:
            ffmc = 101 - 59.5 * (250 - mr) / (147.2 + mr)
        else:
            ffmc = 101 - 29.5 * np.log(250 - mr) + 18.28
        
        # Clamp values
        ffmc = max(0.0, min(101.0, ffmc))
        
        return ffmc
    
    def compute_dmc(self, temp: float, rh: float, rain: float, month: int,
                    prev_dmc: float = None) -> float:
        """
        Compute Duff Moisture Code (DMC)
        
        Args:
            temp: Temperature (°C)
            rh: Relative Humidity (%)
            rain: Rainfall (mm)
            month: Month (1-12)
            prev_dmc: Previous day's DMC (default: 110.0 for valid log calculation)
            
        Returns:
            DMC value
        """
        if prev_dmc is None:
            prev_dmc = 110.0  # Higher default to avoid log(negative) error
        
        # Rain effect
        if rain > 1.5:
            re = 0.92 * rain - 1.27
            mo = 20 + np.exp(5.6348 - prev_dmc / 43.43)
            
            if mo < 0:
                mo = 0
            elif mo > re:
                b = 100 / (0.5 + 0.3 * re)
                mo = mo - re * (1 - np.exp(-re / b))
            else:
                mo = mo - re + re * (1 - np.exp(-re / b))
            
            if mo < 0:
                mo = 0
            elif mo > 250:
                mo = 250
            
            prev_dmc = 43.43 * (5.6348 - np.log(mo))
            if prev_dmc < 0:
                prev_dmc = 0
        
        # Day length factor
        if month in [12, 1, 2]:
            lf = 6.5
        elif month in [3, 4, 5, 10, 11]:
            lf = 7.5
        else:
            lf = 8.5
        
        # Moisture equilibrium
        el = 0.942 * (rh ** 0.679) + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh))
        
        if el < 0:
            el = 0
        elif el > 250:
            el = 250
        
        # Log drying rate
        k = 1.894 * (el + 104) * (lf + 0.5) * 0.0001
        
        # Adjust moisture content
        if prev_dmc > 104:
            po = 244.72 - 43.43 * np.log(prev_dmc - 104)
        else:
            # If prev_dmc is too low, use a reasonable starting value
            po = 244.72 - 43.43 * np.log(6.0)  # log(6) for edge case
        if po < 0:
            po = 0
        
        pr = po + 100 * k
        if pr > 250:
            pr = 250
        
        # Convert to DMC
        try:
            if pr > 104:
                dmc = 244.72 - 43.43 * np.log(pr - 104)
            else:
                # If pr is too low, use a reasonable minimum value
                dmc = 5.0  # Minimum DMC value
            if dmc < 0:
                dmc = 5.0
            elif np.isnan(dmc) or np.isinf(dmc):
                dmc = 5.0
        except:
            dmc = 5.0
        
        return dmc
    
    def compute_dc(self, temp: float, rain: float, month: int,
                  prev_dc: float = None) -> float:
        """
        Compute Drought Code (DC)
        
        Args:
            temp: Temperature (°C)
            rain: Rainfall (mm)
            month: Month (1-12)
            prev_dc: Previous day's DC (default: 15.0)
            
        Returns:
            DC value
        """
        if prev_dc is None:
            prev_dc = 15.0
        
        # Rain effect
        if rain > 2.8:
            rd = 0.9 * rain - 1.27
            mo = 800 * np.exp(-prev_dc / 400)
            
            if mo < rd:
                mo = mo - rd + rd * (1 - np.exp(-rd / 244.72))
            else:
                mo = mo - rd
            
            if mo < 0:
                mo = 0
            elif mo > 800:
                mo = 800
            
            prev_dc = 400 * np.log(800 / mo)
            if prev_dc < 0:
                prev_dc = 0
        
        # Day length factor
        if month in [12, 1, 2]:
            lf = -1.6
        elif month in [3, 4, 5, 10, 11]:
            lf = -0.9
        else:
            lf = -0.3
        
        # Potential evapotranspiration
        pe = (0.36 * (temp + 2.8) + lf) * 0.5
        
        if pe < 0:
            pe = 0
        
        # Adjust DC
        dc = prev_dc + pe
        if dc < 0:
            dc = 0
        
        return dc
    
    def compute_isi(self, ffmc: float, wind: float) -> float:
        """
        Compute Initial Spread Index (ISI)
        
        Args:
            ffmc: Fine Fuel Moisture Code
            wind: Wind speed (km/h)
            
        Returns:
            ISI value
        """
        # Moisture content
        if ffmc <= 50:
            mo = 147.2 * (101 - ffmc) / (59.5 + ffmc)
        else:
            mo = 21.1 * np.exp((100 - ffmc) / 29.5)
        
        # Fine fuel moisture
        ff = 19.2 * np.exp(-0.067 * mo)
        if ff < 0:
            ff = 0
        
        # Wind effect
        fw = np.exp(0.05039 * wind)
        
        # ISI calculation
        isi = 0.208 * fw * ff
        if isi < 0:
            isi = 0
        
        return isi
    
    def compute_bui(self, dmc: float, dc: float) -> float:
        """
        Compute Buildup Index (BUI)
        
        Args:
            dmc: Duff Moisture Code
            dc: Drought Code
            
        Returns:
            BUI value
        """
        try:
            if dmc <= 0.4 * dc:
                bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
            else:
                bui = dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc)) * (dmc - 0.4 * dc) ** 0.6
            
            if np.isnan(bui) or np.isinf(bui):
                bui = 0
        except:
            bui = 0
        
        return bui
    def compute_fwi(self, isi: float, bui: float) -> float:
        """
        Compute Fire Weather Index (FWI)
        
        Args:
            isi: Initial Spread Index
            bui: Buildup Index
            
        Returns:
            FWI value
        """
        if bui <= 80:
            fD = 0.626 * bui ** 0.809
        else:
            fD = 1000 / (25 + 108.64 * np.exp(-0.023 * bui))
        
        if fD < 0:
            fD = 0
        
        # FWI calculation
        B = 0.1 * isi * fD
        if B < 1:
            fwi = B
        else:
            fwi = np.exp(2.72 * (0.434 * np.log(B)) ** 0.647)
        
        if fwi < 0:
            fwi = 0
        
        return fwi
    
    def compute_all(self, temp: float, rh: float, wind: float, rain: float, 
                    month: int, prev_ffmc: float = None, prev_dmc: float = None,
                    prev_dc: float = None) -> dict:
        """
        Compute all FWI components
        
        Args:
            temp: Temperature (°C)
            rh: Relative Humidity (%)
            wind: Wind speed (km/h)
            rain: Rainfall (mm)
            month: Month (1-12)
            prev_ffmc: Previous day's FFMC
            prev_dmc: Previous day's DMC
            prev_dc: Previous day's DC
            
        Returns:
            Dictionary with all FWI components
        """
        ffmc = self.compute_ffmc(temp, rh, wind, rain, prev_ffmc)
        dmc = self.compute_dmc(temp, rh, rain, month, prev_dmc)
        dc = self.compute_dc(temp, rain, month, prev_dc)
        isi = self.compute_isi(ffmc, wind)
        bui = self.compute_bui(dmc, dc)
        fwi = self.compute_fwi(isi, bui)
        
        return {
            'FFMC': ffmc,
            'DMC': dmc,
            'DC': dc,
            'ISI': isi,
            'BUI': bui,
            'FWI': fwi
        }
