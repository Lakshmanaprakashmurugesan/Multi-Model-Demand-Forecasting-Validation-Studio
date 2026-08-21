# Telecommunications Applicability Without Changing the Core Code

## Current application boundary
The application operates on one selected time series at a time. It does not automatically loop across hundreds of SKUs or regions.

## Telecom-compatible use pattern
A telecommunications planner or evaluator can prepare one authorized series representing a specific planning slice, for example:

- one device model in one region;
- one SIM category in one sales channel;
- one network equipment family at one stocking location;
- one repair-part group for one operational area.

That filtered series can be uploaded to the existing application with optional numeric drivers.

## Committed synthetic example
`sample_data/synthetic_telecom_equipment_demand.csv` represents a fictional telecommunications-equipment demand series and contains no employer or carrier data.

Suggested field interpretation:

| Field | Meaning |
|---|---|
| `date` | daily planning period |
| `demand` | synthetic demand quantity |
| `promotion` | synthetic demand-event flag |
| `price_index` | synthetic relative price index |
| `availability_index` | synthetic product availability signal |
| `deployment_activity` | synthetic infrastructure/deployment activity index |

## Broader endeavor connection
The current repository can serve as evidence of a forecasting-validation subsystem and current technical progress. It should not be represented as the complete national telecommunications platform, MEIO engine, or IMEI provenance subsystem.
