# Synthetic Telecommunications Sample Data

`synthetic_telecom_equipment_demand.csv` is deterministic, fictional demonstration data generated with random seed 42.

It does not contain employer, customer, carrier, OEM, or individual data and must not be described as operational telecom data.

The file is intentionally compatible with the unchanged application:

- timestamp: `date`
- target: `demand`
- optional numeric drivers: `promotion`, `price_index`, `availability_index`, `deployment_activity`

The sample exists to make software behavior reproducible. Any forecast accuracy obtained on this file applies only to this synthetic series.
