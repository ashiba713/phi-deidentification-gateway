import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("Patient Alice Morgan visited 14 Oak Street, Boston on 04/12/2025.",
     [("Alice Morgan", "PERSON"), ("14 Oak Street", "LOCATION"),
      ("Boston", "LOCATION"), ("04/12/2025", "DATE")]),

    ("Dr. Robert Wilson treated Mary Jones at Green Valley Hospital.",
     [("Dr. Robert Wilson", "PERSON"), ("Mary Jones", "PERSON"),
      ("Green Valley Hospital", "FACILITY")]),

    ("Patient David Brown, age 94, called 9876543210.",
     [("David Brown", "PERSON"), ("age 94", "AGE_OVER_89"),
      ("9876543210", "PHONE")]),

    ("Contact Emily Davis at emily.davis@example.com.",
     [("Emily Davis", "PERSON"), ("emily.davis@example.com", "EMAIL")]),

    ("MRN: AB12345678 belongs to patient Michael Scott.",
     [("MRN: AB12345678", "MRN"), ("Michael Scott", "PERSON")]),

    ("Patient Sarah Lee visited on March 15, 2025.",
     [("Sarah Lee", "PERSON"), ("March 15, 2025", "DATE")]),

    ("Call fax 2125550199 for patient James Miller.",
     [("2125550199", "PHONE"), ("James Miller", "PERSON")]),

    ("Patient Linda Green lives at 55 Lake Road, Austin, Texas.",
     [("Linda Green", "PERSON"), ("55 Lake Road", "LOCATION"),
      ("Austin", "LOCATION")]),

    ("SSN 123-45-6789 was recorded for Daniel White.",
     [("123-45-6789", "SSN"), ("Daniel White", "PERSON")]),

    ("Patient aged 88 visited on 09/10/2024.",
     [("09/10/2024", "DATE")]),

    ("Follow-up for 3 days after surgery for patient Peter Adams.",
     [("Peter Adams", "PERSON")]),

    ("Member ID: HP123456789 for patient Olivia Martin.",
     [("Member ID: HP123456789", "HEALTH_PLAN_ID"),
      ("Olivia Martin", "PERSON")]),

    ("Account Number: 7788990011 belongs to George King.",
     [("Account Number: 7788990011", "ACCOUNT_ID"), ("George King", "PERSON")]),

    ("Driver's License: DL99887766 belongs to Susan Clark.",
     [("Driver's License: DL99887766", "LICENSE_ID"), ("Susan Clark", "PERSON")]),

    ("Patient William Turner visited on 2025-06-18.",
     [("William Turner", "PERSON"), ("2025-06-18", "DATE")]),

    ("Contact nurse Rachel Adams at 9000011111 or rachel@test.org.",
     [("Rachel Adams", "PERSON"), ("9000011111", "PHONE"),
      ("rachel@test.org", "EMAIL")]),

    ("Patient Henry Ford, age 91, admitted to City Care Hospital.",
     [("Henry Ford", "PERSON"), ("age 91", "AGE_OVER_89"),
      ("City Care Hospital", "FACILITY")]),

    ("Visit record for Emma Watson at 221B Baker Street on 12/01/2025.",
     [("Emma Watson", "PERSON"), ("221B Baker Street", "LOCATION"),
      ("12/01/2025", "DATE")]),

    ("Patient Mark Taylor has IP 192.168.1.20 and record MRN: XY123456.",
     [("Mark Taylor", "PERSON"), ("192.168.1.20", "IP_ADDRESS"),
      ("MRN: XY123456", "MRN")]),

    ("Patient Sophia Hall is 90 years old. Appointment in 2 weeks.",
     [("Sophia Hall", "PERSON"), ("90 years old", "AGE_OVER_89")]),
]


def main():
    records = []

    for index, (text, entities) in enumerate(CASES, start=1):
        labelled = []

        for value, label in entities:
            start = text.find(value)

            if start == -1:
                raise ValueError(
                    f"Entity not found in note {index}: {value}"
                )

            labelled.append({
                "start": start,
                "end": start + len(value),
                "text": value,
                "entity_type": label
            })

        records.append({
            "id": index,
            "text": text,
            "entities": labelled
        })

    output = ROOT / "data" / "synthetic_test.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Created {len(records)} labelled test notes.")
    print(f"Ground-truth entities: {sum(len(x['entities']) for x in records)}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
