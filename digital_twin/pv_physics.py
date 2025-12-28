def pv_physical_model(irradiance, temperature):
    temp_coeff = -0.004
    return irradiance * (1 + temp_coeff * (temperature - 25))
