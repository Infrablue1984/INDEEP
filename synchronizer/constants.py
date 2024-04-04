"""This module contains all global string variable of the model.

Typical usage example:

    import synchronizer.constants as cs/con/c/co
"""

# Disease course statuses
SUSCEPTIBLE = "susceptible"
EXPOSED = "exposed"
ASYMPTOMATIC = "asymptomatic"
PRESYMPTOMATIC = "presymptomatic"
MILD = "mild"
SEVERE = "severe"
CRITICAL = "critical"
RECOVERED = "recovered"
DEAD = "dead"

# Epidemic properties
INFECTED = "infected"
INFECTIOUS = "infectious"
SYMPTOMATIC = "symptomatic"
NEXT_STATUS_SET = "next_status_set"

# Public statuses
VACCINATED = "vaccinated"
HOSPITALIZED = "hospitalized"
QUARANTINED = "quarantined"
DIAGNOSED = "diagnosed"
ISOLATED = "isolated"
MASK_WEARING = "mask_wearing"
DO_SOCIAL_DISTANCING = "social_distancing"
FEEL_BAD = "feel_bad"

# Public temporary properties
TESTED = "tested"
POS_TESTED = "pos_tested"
NEG_TESTED = "neg_tested"
TEST_RESULTS = "test_results"
LABORATORY_TEST = "laboratory_test"
SENSITIVITY = "test_sensitivity"
SPECIFICITY = "test_specificity"

# General properties (agent data)
ID = "id"
AGE = "age"
Z_CODE = "z_code"
OCCUPATION = "occupation"
SEX = "sex"
LANDSCAPE = "landscape"
SOCIABLE = "sociable"

# general names
RAPID_TEST = "rapid_test"
TEST_INFO = "test_info"
PERCENTAGE_RAPID_TEST = "percentage_rapid_test"
PRIMARY = "primary"
CLUSTER = "cluster"
PERIOD = "period"
TIME_DELAY = "time_delay"
PERCENTAGE_MILD_FEEL_BAD = "percentage_mild_feel_bad"
ALL = "all"
RANDOM = "random"
PRIMARY_CONTACT = "primary contact"
PUPIL_TEACHER = "pupil/teacher"
BIGGER_65 = "> 65"

# general
AGE_GR = "Age_gr"
MIN = "Min"
MAX = "Max"
DATE = "date"
AGE_GROUP = "age_group"
PARAMETER = "parameter"
VALUE = "value"
TYPE = "type"
# TODO: Decide what to put here

# network attributes
PARENT_ID = "parent id"
NET_ID = "id"
TYPE = "type"
SIZE = "size"
MEAN_CONTACTS = "mean contacts"
YEAR = "year"
WEEKDAY = "weekday"
LANDSCAPE = "landscape"
LOCALIZED = "localized"
CODE = "code"
STATUS = "status"
SUB_GR = "sub_gr"

# network types
HOUSEHOLDS = "households"
WORKPLACES = "workplaces"
KITAS = "kitas"
SCHOOLS = "schools"
UNIVERSITIES = "universities"
ACTIVITIES = "activities"
GEOGRAPHIC = "geographic"

# local types
OFFICE = "office"
MEETING = "meeting"
CLASS = "class"
LECTURE = "lecture"
FREE = "free"
MENSA = "mensa"
UNSPECIFIC = "unspecific"
ACTIVITY = "activity"
LIVING_ROOM = "living room"

# room attributes
ROOM_VOLUME = "room_volume"
AIR_EXCHANGE_RATE = "air_exchange_rate"
INHALATION_RATE = "inhalation_rate"
CONTACT_HOURS = "contact_hours"

TAG = " "
# intervention names --> translation user input and programme
LIMIT_EVENT_SIZE = "limit event size"
LIMIT_EVENT_SIZE_WORKPLACES = f"{LIMIT_EVENT_SIZE}{TAG}{WORKPLACES}"
LIMIT_EVENT_SIZE_KITAS = f"{LIMIT_EVENT_SIZE}{TAG}{KITAS}"
LIMIT_EVENT_SIZE_SCHOOLS = f"{LIMIT_EVENT_SIZE}{TAG}{SCHOOLS}"
LIMIT_EVENT_SIZE_UNIVERSITIES = f"{LIMIT_EVENT_SIZE}{TAG}{UNIVERSITIES}"
LIMIT_EVENT_SIZE_ACTIVITIES = f"{LIMIT_EVENT_SIZE}{TAG}{ACTIVITIES}"
SPLIT_GROUPS = "split groups"
SPLIT_GROUPS_WORKPLACES = f"{SPLIT_GROUPS}{TAG}{WORKPLACES}"
SPLIT_GROUPS_KITAS = f"{SPLIT_GROUPS}{TAG}{KITAS}"
SPLIT_GROUPS_SCHOOLS = f"{SPLIT_GROUPS}{TAG}{SCHOOLS}"
SPLIT_GROUPS_UNIVERSITIES = f"{SPLIT_GROUPS}{TAG}{UNIVERSITIES}"
SPLIT_GROUPS_ACTIVITIES = f"{SPLIT_GROUPS}{TAG}{ACTIVITIES}"
HOME_DOING = "home doing"
HOME_DOING_WORKPLACES = f"{HOME_DOING}{TAG}{WORKPLACES}"
HOME_DOING_KITAS = f"{HOME_DOING}{TAG}{KITAS}"
HOME_DOING_SCHOOLS = f"{HOME_DOING}{TAG}{SCHOOLS}"
HOME_DOING_UNIVERSITIES = f"{HOME_DOING}{TAG}{UNIVERSITIES}"
HOME_DOING_ACTIVITIES = f"{HOME_DOING}{TAG}{ACTIVITIES}"
SHUT_INSTITUTIONS = "shut institutions"
SHUT_WORKPLACES = f"{SHUT_INSTITUTIONS}{TAG}{WORKPLACES}"
SHUT_SCHOOLS = f"{SHUT_INSTITUTIONS}{TAG}{SCHOOLS}"
SHUT_KITAS = f"{SHUT_INSTITUTIONS}{TAG}{KITAS}"
SHUT_UNIVERSITIES = f"{SHUT_INSTITUTIONS}{TAG}{UNIVERSITIES}"
SHUT_ACTIVITIES = f"{SHUT_INSTITUTIONS}{TAG}{ACTIVITIES}"
SOCIAL_DISTANCING = "social distancing"
MASK = "mask"
CLUSTER_ISOLATION = "cluster isolation"
MANUAL_CONTACT_TRACING = "manual contact tracing"
SELF_ISOLATION = "self isolation"
TESTING = "testing"


# class names

PUBLIC_REGULATOR = "PublicRegulator"
NETWORK_MANAGER = "NetworkManager"
EPIDEMIC_SPREADER = "EpidemicSpreader"

GERMAN = "german"
ENGLISH = "english"
START_END_DATE_EXCEPTION = "Start end date exception"
CHECK_INTERVENTION_EXCEPTION = "Check intervention exception"
INTERVENTION_DATE_OVERLAP_EXCEPTION = "intervention date overlap exception"
INTERVENTION_SUPER_SPECIFIC_OVERLAP_EXCEPTION = (
    "intervention super specific overlap exception"
)

LANGUAGE_DICT = {
    START_END_DATE_EXCEPTION: {
        ENGLISH: "The start date must be before end date.",
        GERMAN: "Das Anfangsdatum muss vor dem Enddatum liegen.",
    },
    CHECK_INTERVENTION_EXCEPTION: {
        ENGLISH: "Please check all interventions.",
        GERMAN: "Bitte überprüfen Sie alle Maßnahmen.",
    },
    INTERVENTION_DATE_OVERLAP_EXCEPTION: {
        ENGLISH: (
            "You have overlapping interventions of the same type. \nThe start date "
            "of one choice must be greater-than-or-equal the end date of another."
        ),
        GERMAN: (
            "Es überlappen sich Anfangs- und Enddatum gleicher Maßnahmen. \nBitte"
            " stellen Sie sicher, dass dasAnfangsdatum der einen Auswahl größergleich"
            " dem Enddatum einer anderen Auswahl ist."
        ),
    },
    INTERVENTION_SUPER_SPECIFIC_OVERLAP_EXCEPTION: {
        ENGLISH: (
            "You have overlapping interventions of the same type. \nPlease note: "
            "Some interventions do imply others."
        ),
        GERMAN: (
            "Es überlappen sich Anfangs- und Enddatum gleicher Maßnahmen. \nBitte"
            " beachten Sie, dass einige Maßnahmen bereits andere beinhalten."
        ),
    },
}
