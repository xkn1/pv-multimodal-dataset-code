# Power-data time semantics

The source power workbooks contain one reported power value in kW for each timestamp at an expected five-minute cadence.

The release pipeline:

1. reads the source timestamp and power value directly;
2. converts the timestamp and value to typed fields;
3. applies auditable quality flags;
4. does not resample, average, sum or interpolate power values; and
5. pairs each image to the nearest power timestamp within 30 seconds.

Therefore the public dataset describes these records as "source-reported power measurements at five-minute timestamps." It does not claim that a value is a five-minute mean, interval energy, or instantaneous meter reading because that upstream platform semantic was not independently verified.

The 15-minute ramp calculation uses differences between the retained five-minute timestamped power values.
