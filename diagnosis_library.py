class DiagnosisLibrary:
    def __init__(self):
        self.classes = ['Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma', 
                        'Atopic Dermatitis', 'Bullous Disease', 'Cellulitis Impetigo', 
                        'Eczema', 'Exanthems', 'Alopecia', 'Herpes HPV', 
                        'Pigmentation Light Disease', 'Lupus', 'Melanoma', 
                        'Nail Fungus', 'Poison Ivy', 'Psoriasis', 'Scabies Lyme Disease', 
                        'Seborrheic Keratoses (benign tumor)', 'Systemic Disease', 
                        'Tinea Ringworm Candidiasis', 'Urticaria Hives', 
                        'Vascular Tumors', 'Vasculitis', 'Warts Molluscum']

    def get_description(self, class_name):
        name = ''.join(class_name.split())
        path = "./information_on_classes/description/" + name + ".txt"
        with open(path, 'r', encoding='utf-8') as file:
            data = file.read()
            return data

    def get_symptoms(self, class_name):
        name = ''.join(class_name.split())
        path = "./information_on_classes/symptoms/" + name + ".txt"
        with open(path, 'r', encoding='utf-8') as file:
            data = file.read()
            return data


    def get_course_of_actions(self, class_name):
        name = ''.join(class_name.split())
        path = "./information_on_classes/recommended_course_of_action/" + name + ".txt"
        with open(path, 'r', encoding='utf-8') as file:
            data = file.read()
            return data

def api_get_text(name):
    self = DiagnosisLibrary()
    description = self.get_description(name)
    symptoms = self.get_symptoms(name)
    coa = self.get_course_of_actions(name)

    return description, symptoms, coa


def test_suite():
    description, symptoms, coa = api_get_text('Acne and Rosacea')
    a = ""

if __name__ == "__main__":
    test_suite()