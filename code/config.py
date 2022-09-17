
from dynaconf import Dynaconf

### RQ 1 CONFIG ####


collection_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_collection.toml']
)

other_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_other.toml']
)

sharing_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_sharing.toml']
)

processing_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_processing.toml']
)

functionality_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_functionality.toml']
)

purpose_other_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_other.toml']
)

purpose_advertisement = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_advertisement.toml']
)

purpose_analytics = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_analytics.toml']
)


### RQ 2 CONFIG ####

## Practice ##

# Collection
collection_1_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_1_hop_collection.toml']
)

collection_2_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_2_hop_collection.toml']
)

collection_3_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_3_hop_collection.toml']
)

# Sharing

sharing_1_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_1_hop_sharing.toml']
)

sharing_2_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_2_hop_sharing.toml']
)

sharing_3_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_3_hop_sharing.toml']
)

# Processing

processing_1_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_1_hop_processing.toml']
)

processing_2_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_2_hop_processing.toml']
)

processing_3_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_3_hop_processing.toml']
)

# Other

practice_other_1_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_1_hop_other.toml']
)

practice_other_2_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_2_hop_other.toml']
)

practice_other_3_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/practice_3_hop_other.toml']
)

## Purpose ##

# Advertisement

advertisement_1_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_1_hop_advertisement.toml']
)

advertisement_2_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_2_hop_advertisement.toml']
)

advertisement_3_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_3_hop_advertisement.toml']
)

# Analytics

analytics_1_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_1_hop_analytics.toml']
)

analytics_2_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_2_hop_analytics.toml']
)

analytics_3_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_3_hop_analytics.toml']
)

# Functionality

functionality_1_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_1_hop_functionality.toml']
)

functionality_2_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_2_hop_functionality.toml']
)

functionality_3_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_3_hop_functionality.toml']
)

# Other

purpose_other_1_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_1_hop_other.toml']
)

purpose_other_2_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_2_hop_other.toml']
)

purpose_other_3_hop_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/purpose_3_hop_other.toml']
)
