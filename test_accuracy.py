# test_accuracy.py
# Evaluate intent classification for the refined Hep B Chatbot
#
# Run with:
#   python test_accuracy.py
#
# Assumes HepBChat_refined.py is in the same folder and defines classify_intent_v2.

from HepBChat_refined import classify_intent_v2 as classify_intent

# 50 labeled queries across 10 intents (5 each)
# Intents: greeting, transmission, symptoms, testing, vaccination,
#          prevention, treatment, window, lab_markers, urgent

TEST_DATA = [
    # greeting (5)
    ("hello there", "greeting"),
    ("hey can you help me", "greeting"),
    ("hi, I need some help", "greeting"),
    ("good afternoon, hello", "greeting"),
    ("hey there, can you start", "greeting"),

    # transmission (5) 
    ("how does hepatitis b spread from person to person", "transmission"),
    ("is hepatitis b contagious through blood and body fluids", "transmission"),
    ("can sharing needles transmit hepatitis b infection", "transmission"),
    ("can I catch hep b by borrowing a razor from someone", "transmission"),  
    ("can you get hep b from unprotected sex or kissing", "transmission"),    

    # symptoms (5) 
    ("what are common symptoms of hepatitis b infection", "symptoms"),
    ("do yellow eyes and jaundice mean I might have hep b", "symptoms"),
    ("does hep b cause fatigue, nausea and feeling sick", "symptoms"),
    ("why do I feel tired all the time with dark urine", "symptoms"),         
    ("do I need to worry about liver pain and nausea with possible hep b", "symptoms"),  

    # testing (5) 
    ("what blood tests are used to check for hepatitis b", "testing"),
    ("should I get hepatitis b screening blood work", "testing"),
    ("which labs and tests check for hep b infection", "testing"),
    ("is the HBsAg blood test the main test for hepatitis b", "testing"),
    ("I might have been exposed, what hepatitis b test should I ask for", "testing"), 

    # vaccination (5) 
    ("should I get the hepatitis b vaccine to protect myself", "vaccination"),
    ("how many doses are in the hep b vaccine series", "vaccination"),
    ("is there a vaccine shot that prevents hepatitis b", "vaccination"),
    ("what is the recommended hepatitis b immunization schedule for adults", "vaccination"),
    ("if I was vaccinated as a kid, am I still protected from hep b", "vaccination"),    

    # prevention (5)
    ("how can I prevent hepatitis b infection", "prevention"),
    ("do condoms help reduce the risk of getting hep b", "prevention"),
    ("should I avoid sharing needles to prevent hepatitis b", "prevention"),
    ("is sharing razors risky and should I avoid it for hep b prevention", "prevention"),
    ("what safer behaviors can lower my risk of hepatitis b", "prevention"), 

    # treatment (5) 
    ("is there treatment available for chronic hepatitis b", "treatment"),
    ("what antiviral medicines are used to treat hepatitis b", "treatment"),
    ("can tenofovir medication be used to treat chronic hep b", "treatment"),
    ("how is chronic hepatitis b usually managed over time", "treatment"),
    ("besides medicine, what can I do to manage my hepatitis b", "treatment"),  

    # window (5) 
    ("what is the window period for hepatitis b testing after exposure", "window"),
    ("how long after exposure should I wait before testing for hep b", "window"),
    ("can a hep b test be negative during the early window period", "window"),
    ("I was exposed last week, when should I get tested for hepatitis b", "window"),    
    ("how long does it take for hepatitis b to show up on blood work", "window"),

    # lab_markers (5) 
    ("what does the lab marker HBsAg mean on my hepatitis b test", "lab_markers"),
    ("what is the difference between anti-HBs and anti-HBc antibodies", "lab_markers"),
    ("what does HBeAg indicate in chronic hepatitis b", "lab_markers"),
    ("does HBV DNA represent the viral load for hepatitis b", "lab_markers"),
    ("which hepatitis b antibodies show that I am immune now", "lab_markers"),

    # urgent (5)
    ("I am vomiting blood after hep b exposure, is this an emergency", "urgent"),
    ("black stools and severe abdominal pain, should I go to the ER", "urgent"),
    ("I am pregnant and think I was exposed to hep b, do I need urgent care", "urgent"),
    ("confusion and jaundice with high fever, is this urgent", "urgent"),
    ("I feel very sick with strong stomach pain after hepatitis b, should I seek emergency help", "urgent"),
]


def evaluate():
    total = len(TEST_DATA)
    correct = 0
    per_intent_correct = {}
    per_intent_total = {}
    mistakes = []

    for question, expected in TEST_DATA:
        predicted, conf = classify_intent(question)

        per_intent_total[expected] = per_intent_total.get(expected, 0) + 1

        if predicted == expected:
            correct += 1
            per_intent_correct[expected] = per_intent_correct.get(expected, 0) + 1
        else:
            mistakes.append((question, expected, predicted, conf))
            per_intent_correct.setdefault(expected, 0)

    overall_acc = 100.0 * correct / total
    print(f"Overall accuracy: {overall_acc:.2f}%  ({correct}/{total})\n")

    print("Per-intent accuracy:")
    for intent in sorted(per_intent_total.keys()):
        ok = per_intent_correct.get(intent, 0)
        cnt = per_intent_total[intent]
        acc = 100.0 * ok / cnt
        print(f"  - {intent:12s}: {ok:2d}/{cnt:2d}  ({acc:5.1f}%)")
    print()

    if mistakes:
        print("Misclassifications:")
        for q, exp, pred, conf in mistakes:
            print(f"  Q: {q!r}\n"
                  f"     expected = {exp}, predicted = {pred}, confidence = {conf:.2f}\n")
    else:
        print("No misclassifications. Nice!")


if __name__ == "__main__":
    evaluate()


