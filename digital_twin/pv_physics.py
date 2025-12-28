def pv_power(irradiance, temperature, efficiency=0.18):
    temp_coeff = 1 - 0.004 * (temperature - 25)
    return irradiance * efficiency * temp_coeff
