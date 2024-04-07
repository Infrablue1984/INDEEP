# COVID_default.csv

### General information

The COVID_default.csv files contains the epidemiological parameteres for the spread of covid19, which in our model accounts for the so called AirborneVirusSpreader's. (**Attention: the parameters are not validated yet, even not for covid19!!!**).

INDEEP is also meant to model other airborne spreaded diseases, e.g. influenza, with identical parameter types, but different values. Those values have not been researched yet.

INDEEP is also meant to model other type of disease, e.g. smear infections, which is not implemented yet, but can be easily extended.

The parameter description is incomplete due to limited time to translate the documentation from German to English.

### Parameter adjustment

1. Go to the line of the parameter to adjust.
2. Change the number of the value. (Do not change the name of the value or the file structure, this will raise an error!)
3. Save file.

### Parameter Description

###### Unreported factor

The unreported factor describes the disproportion of diagnosed (reported) cases of illness to the number of citizens who are actually ill.An unreported number of 3 corresponds to the situation of three times as many sick people as are actually reported.

###### reported_infected_percentage

###### reported_infected_percentage

###### initially_recovered_percentage

###### initially_vaccinated_percentage

###### hospital_percentage

###### viral_load_mu and viral_load_sigma

The viral load describes the amount of viruses found in the saliva of an infected person and is given in this model as the concentration $\log_{10}(\text{RNA Kopien pro mL})$. Within a population, the viral load can vary greatly from person to person, but also within the course of the disease.
In this model, each agent is therefore assigned a standard viral load, which is determined using a logarithmic normal distribution.
This distribution is determined by two parameters $\mu$ and $\sigma$, which can be set separately in the tool. As is usual in medicine, the unit of $\log_{10}(\text{RNA Kopien pro mL})$ is used.
At the start of an agent's infectivity, its viral load is increased for 4 days by doubling its standard viral load.

###### quanta_conversion_factor

###### droplet_volume_concentration_mean

###### droplet_volume_concentration_std

###### gamma and beta

In order to make infection via direct contact consistent with infection via aerosols, two abstract parameters $\gamma$ and $\beta$ were introduced. Following the approach of (Buonanno et. Al) for determining the inhaled infectious quanta, the model includes the calculation using  dose of received quanta. This is calculated using the following formula: $D_q = f_s \cdot f_m \cdot IR_e \cdot IR_r \cdot QEL \cdot \gamma \int_{0}^{T}\left(1-e^{-\beta\cdot t}\right) dt$ , where $f_s$ and $f_m$ include keeping your distance and wearing a mask as factors, $IR_e$ and $IR_r$ describe the inhalation rate of the emitter and receiver, $QEL$ the quantum emission load and $T$ indicates the duration of the contact.
Since the parameters $\gamma$ and $\beta$ have no physical meaning, they must be calibrated for a reliable simulation.

###### particle_deposition

The deposition rate of aerosols is given by
\begin{align*}
    k = \frac{1}{t_d},
\end{align*}
$k = \frac{1}{t_d}$, where the deposition time $t_d$ of an aerosol is determined as the quotient of the deposition speed and the height of the emission source. The unit used in the model is $h^{-1}$.

###### viral_inactivation

The inactivation of viruses is described by the exponential decay $    N(t) = N_0e^{-kt}$,
where $N_0$ represents the number of viruses at the beginning.
The virus inactivation rate (or general elimination constant) $k$ can be expressed by the half-life $t_{1/2}$ as follows
$k = \frac{\ln 2}{t_{1/2}}$. The virus inactivation rate is given in the unit $h^{-1}$.

###### mask_factor_emitter

###### mask_factor_receiver

###### social_distancing_factor

###### vaccination_factor_viral_load

###### vaccination_factor_relative_sensitivity

###### vaccination_factor_progression

###### not_hospitalized_factor

###### exposed_to_asymptomatic_mean

###### exposed_to_presymptomatic_mean

###### asymptomatic_to_recovered_mean

###### presymptomatic_to_mild_mean

###### mild_to_severe_mean

###### severe_to_critical_mean

###### critical_to_dead_mean

###### mild_to_recovered_mean
###### severe_to_recovered_mean

###### critical_to_recovered_mean
###### exposed_to_asymptomatic_std

###### exposed_to_presymptomatic_std

###### asymptomatic_to_recovered_std

###### presymptomatic_to_mild_std

###### mild_to_severe_std

###### severe_to_critical_std

###### critical_to_dead_std

###### mild_to_recovered_std

###### severe_to_recovered_std

###### critical_to_recovered_std

###### symptomatic_probability_from_0 To symptomatic_probability_from_90
###### severe_probability_from_0 to
severe_probability_from_90
###### critical_probability_from_0 to critical_probability_from_90
###### dead_probability_from_0 to dead_probability_from_90
