import json
import random
import re
from pathlib import Path
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_train.json"

def make_note():
    name = fake.name()
    city = fake.city()
    facility = fake.company() + " Hospital"
    date = fake.date_between(start_date="-5y", end_date="today").strftime("%m/%d/%Y")
    email = fake.email()
    phone = fake.phone_number()
    mrn = "MRN" + str(random.randint(100000, 999999))
    age = random.randint(90, 105)

    templates = [
        (
            f"Patient {name} visited {facility} in {city} on {date}. "
            f"Age {age}. Contact {phone} or {email}. Record {mrn}.",
            [name, facility, city, date, email, phone, mrn, f"Age {age}"]
        ),
        (
            f"Dr. {name} reviewed the patient record at {facility}. "
            f"The patient lives in {city}. Appointment date {date}. "
            f"Call {phone}. Email {email}.",
            [f"Dr. {name}", facility, city, date, phone, email]
        ),
        (
            f"{name} was admitted to {facility} on {date}. "
            f"Patient age {age}. Medical Record Number: {mrn}.",
            [name, facility, date, f"age {age}", f"Medical Record Number: {mrn}"]
        ),
        (
            f"Follow-up for {name} from {city}. "
            f"Contact details: {email}, {phone}. Visit date {date}.",
            [name, city, email, phone, date]
        )
    ]

    text, entities = random.choice(templates)

    # More specific entity spans are assigned first.
    spans = []

    for entity in entities:
        start = text.find(entity)
        if start >= 0:
            if entity.lower().startswith("dr."):
                label = "PERSON"
            elif entity.lower().startswith("age"):
                label = "AGE_OVER_89"
            elif entity.lower().startswith("medical record"):
                label = "MRN"
            elif entity == name or entity == f"Dr. {name}":
                label = "PERSON"
            elif entity == facility:
                label = "FACILITY"
            elif entity == city:
                label = "LOCATION"
            elif entity == date:
                label = "DATE"
            elif entity == email:
                label = "EMAIL"
            elif entity == phone:
                label = "PHONE"
            else:
                label = "MRN"

            spans.append((start, start + len(entity), label))

    tokens = []
    tags = []

    for match in re.finditer(r"\w+|[^\w\s]", text):
        token = match.group()
        start, end = match.span()
        label = None

        for entity_start, entity_end, entity_label in spans:
            if start >= entity_start and end <= entity_end:
                label = entity_label
                break

        if label:
            # BIO tagging: first token B-, remaining tokens I-.
            previous_same = (
                tags and tags[-1] in (f"B-{label}", f"I-{label}")
                and tokens
            )
            if previous_same:
                tags.append(f"I-{label}")
            else:
                tags.append(f"B-{label}")
        else:
            tags.append("O")

        tokens.append(token)

    return {
        "text": text,
        "tokens": tokens,
        "tags": tags
    }


def main():
    records = [make_note() for _ in range(600)]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    print(f"Generated {len(records)} synthetic notes.")
    print(f"Saved to: {OUTPUT}")
    print("\nSample:")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()