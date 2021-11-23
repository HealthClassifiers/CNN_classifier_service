from tensorflow.keras.utils import to_categorical
import glob 
import numpy as np
from PIL import Image

class MedicalDataUtils:
    def convert_images_to_arr(self, images, all):
        res = []

        for img in images:
            img = Image.open(img)
            rezised_img = img.resize((32, 32))
            data = np.asarray(rezised_img)
            res.append(data)

        all.extend(res)

        return res

    def load_images_from_folder(self, folder, all):
        res = []
        path = folder + "/*.jpg"
        for img in glob.glob(path):
            res.append(img)

        return self.convert_images_to_arr(res, all)

    def prep_pixels(self, train_x, train_y, test_x, test_y):
        trainY = to_categorical(train_y)
        testY = to_categorical(test_y)

        # convert from integers to floats
        train_norm = train_x.astype('float32')
        test_norm = test_x.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, trainY, test_norm, testY

    def load_medical_data(self):
        #LOAD TEST DATA
        all_test_x = []
        all_test_y = []
        all_train_x = []
        all_train_y = []

        test_acne_x = self.load_images_from_folder('./skin_disease_data/test/Acne_and_Rosacea_Photos', all_test_x)
        test_acne_y = np.empty(len(test_acne_x))
        test_acne_y.fill(0)
        all_test_y.extend(test_acne_y)

        test_actinic_x = self.load_images_from_folder('./skin_disease_data/test/Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', all_test_x)
        test_actinic_y = np.empty(len(test_actinic_x))
        test_actinic_y.fill(1)
        all_test_y.extend(test_actinic_y)

        test_atopic_x = self.load_images_from_folder('./skin_disease_data/test/Atopic Dermatitis Photos', all_test_x)
        test_atopic_y = np.empty(len(test_atopic_x))
        test_atopic_y.fill(2)
        all_test_y.extend(test_atopic_y)

        test_bullous_x = self.load_images_from_folder('./skin_disease_data/test/Bullous Disease Photos', all_test_x)
        test_bullous_y = np.empty(len(test_bullous_x))
        test_bullous_y.fill(3)
        all_test_y.extend(test_bullous_y)

        test_cellulitis_x = self.load_images_from_folder('./skin_disease_data/test/Cellulitis Impetigo and other Bacterial Infections', all_test_x)
        test_cellulitis_y = np.empty(len(test_cellulitis_x))
        test_cellulitis_y.fill(4)
        all_test_y.extend(test_cellulitis_y)

        test_eczema_x = self.load_images_from_folder('./skin_disease_data/test/Eczema Photos', all_test_x)
        test_eczema_y = np.empty(len(test_eczema_x))
        test_eczema_y.fill(5)
        all_test_y.extend(test_eczema_y)

        test_exanthems_x = self.load_images_from_folder('./skin_disease_data/test/Exanthems and Drug Eruptions', all_test_x)
        test_exanthems_y = np.empty(len(test_exanthems_x))
        test_exanthems_y.fill(6)
        all_test_y.extend(test_exanthems_y)

        test_hair_loss_alopecia_x = self.load_images_from_folder('./skin_disease_data/test/Hair Loss Photos Alopecia and other Hair Diseases', all_test_x)
        test_hair_loss_alopecia_y = np.empty(len(test_hair_loss_alopecia_x))
        test_hair_loss_alopecia_y.fill(1)
        all_test_y.extend(test_hair_loss_alopecia_y)

        test_herpes_x = self.load_images_from_folder('./skin_disease_data/test/Herpes HPV and other STDs Photos', all_test_x)
        test_herpes_y = np.empty(len(test_herpes_x))
        test_herpes_y.fill(7)
        all_test_y.extend(test_herpes_y)

        test_light_diseases_x = self.load_images_from_folder('./skin_disease_data/test/Light Diseases and Disorders of Pigmentation', all_test_x)
        test_light_diseases_y = np.empty(len(test_light_diseases_x))
        test_light_diseases_y.fill(8)
        all_test_y.extend(test_light_diseases_y)

        test_lupus_x = self.load_images_from_folder('./skin_disease_data/test/Lupus and other Connective Tissue diseases', all_test_x)
        test_lupus_y = np.empty(len(test_lupus_x))
        test_lupus_y.fill(9)
        all_test_y.extend(test_lupus_y)

        test_melanoma_x = self.load_images_from_folder('./skin_disease_data/test/Melanoma Skin Cancer Nevi and Moles', all_test_x)
        test_melanoma_y = np.empty(len(test_melanoma_x))
        test_melanoma_y.fill(10)
        all_test_y.extend(test_melanoma_y)

        test_nail_fungus_x = self.load_images_from_folder('./skin_disease_data/test/Nail Fungus and other Nail Disease', all_test_x)
        test_nail_fungus_y = np.empty(len(test_nail_fungus_x))
        test_nail_fungus_y.fill(11)
        all_test_y.extend(test_nail_fungus_y)

        test_poison_ivy_x = self.load_images_from_folder('./skin_disease_data/test/Poison Ivy Photos and other Contact Dermatitis', all_test_x)
        test_poison_ivy_y = np.empty(len(test_poison_ivy_x))
        test_poison_ivy_y.fill(12)
        all_test_y.extend(test_poison_ivy_y)

        test_psoriasis_x = self.load_images_from_folder('./skin_disease_data/test/Psoriasis pictures Lichen Planus and related diseases', all_test_x)
        test_psoriasis_y = np.empty(len(test_psoriasis_x))
        test_psoriasis_y.fill(13)
        all_test_y.extend(test_psoriasis_y)

        test_scabies_lyme_x = self.load_images_from_folder('./skin_disease_data/test/Scabies Lyme Disease and other Infestations and Bites', all_test_x)
        test_scabies_lyme_y = np.empty(len(test_scabies_lyme_x))
        test_scabies_lyme_y.fill(14)
        all_test_y.extend(test_scabies_lyme_y)

        test_seborrheic_keratoses_x = self.load_images_from_folder('./skin_disease_data/test/Seborrheic Keratoses and other Benign Tumors', all_test_x)
        test_seborrheic_keratoses_y = np.empty(len(test_seborrheic_keratoses_x))
        test_seborrheic_keratoses_y.fill(15)
        all_test_y.extend(test_seborrheic_keratoses_y)

        test_systemic_disease_x = self.load_images_from_folder('./skin_disease_data/test/Systemic Disease', all_test_x)
        test_systemic_disease_y = np.empty(len(test_systemic_disease_x))
        test_systemic_disease_y.fill(16)
        all_test_y.extend(test_systemic_disease_y)

        test_tinea_ringworm_x = self.load_images_from_folder('./skin_disease_data/test/Tinea Ringworm Candidiasis and other Fungal Infections', all_test_x)
        test_tinea_ringworm_y = np.empty(len(test_tinea_ringworm_x))
        test_tinea_ringworm_y.fill(17)
        all_test_y.extend(test_tinea_ringworm_y)

        test_urticaria_hives_x = self.load_images_from_folder('./skin_disease_data/test/Urticaria Hives', all_test_x)
        test_urticaria_hives_y = np.empty(len(test_urticaria_hives_x))
        test_urticaria_hives_y.fill(18)
        all_test_y.extend(test_urticaria_hives_y)

        test_vascular_tumors_x = self.load_images_from_folder('./skin_disease_data/test/Vascular Tumors', all_test_x)
        test_vascular_tumors_y = np.empty(len(test_vascular_tumors_x))
        test_vascular_tumors_y.fill(19)
        all_test_y.extend(test_vascular_tumors_y)

        test_vasculitis_x = self.load_images_from_folder('./skin_disease_data/test/Vasculitis Photos', all_test_x)
        test_vasculitis_y = np.empty(len(test_vasculitis_x))
        test_vasculitis_y.fill(20)
        all_test_y.extend(test_vasculitis_y)

        test_warts_molluscum_x = self.load_images_from_folder('./skin_disease_data/test/Warts Molluscum and other Viral Infections', all_test_x)
        test_warts_molluscum_y = np.empty(len(test_warts_molluscum_x))
        test_warts_molluscum_y.fill(21)
        all_test_y.extend(test_warts_molluscum_y)


        #LOAD TRAIN DATA
        train_acne_x = self.load_images_from_folder('./skin_disease_data/train/Acne_and_Rosacea_Photos', all_train_x)
        train_acne_y = np.empty(len(train_acne_x))
        train_acne_y.fill(0)
        all_train_y.extend(train_acne_y)

        train_actinic_x = self.load_images_from_folder('./skin_disease_data/train/Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', all_train_x)
        train_actinic_y = np.empty(len(train_actinic_x))
        train_actinic_y.fill(1)
        all_train_y.extend(train_actinic_y)

        train_atopic_x = self.load_images_from_folder('./skin_disease_data/train/Atopic Dermatitis Photos', all_train_x)
        train_atopic_y = np.empty(len(train_atopic_x))
        train_atopic_y.fill(2)
        all_train_y.extend(train_atopic_y)

        train_bullous_x = self.load_images_from_folder('./skin_disease_data/train/Bullous Disease Photos', all_train_x)
        train_bullous_y = np.empty(len(train_bullous_x))
        train_bullous_y.fill(3)
        all_train_y.extend(train_bullous_y)

        train_cellulitis_x = self.load_images_from_folder('./skin_disease_data/train/Cellulitis Impetigo and other Bacterial Infections', all_train_x)
        train_cellulitis_y = np.empty(len(train_cellulitis_x))
        train_cellulitis_y.fill(4)
        all_train_y.extend(train_cellulitis_y)

        train_eczema_x = self.load_images_from_folder('./skin_disease_data/train/Eczema Photos', all_train_x)
        train_eczema_y = np.empty(len(train_eczema_x))
        train_eczema_y.fill(5)
        all_train_y.extend(train_eczema_y)

        train_exanthems_x = self.load_images_from_folder('./skin_disease_data/train/Exanthems and Drug Eruptions', all_train_x)
        train_exanthems_y = np.empty(len(train_exanthems_x))
        train_exanthems_y.fill(6)
        all_train_y.extend(train_exanthems_y)

        train_hair_loss_alopecia_x = self.load_images_from_folder('./skin_disease_data/train/Hair Loss Photos Alopecia and other Hair Diseases', all_train_x)
        train_hair_loss_alopecia_y = np.empty(len(train_hair_loss_alopecia_x))
        train_hair_loss_alopecia_y.fill(1)
        all_train_y.extend(train_hair_loss_alopecia_y)

        train_herpes_x = self.load_images_from_folder('./skin_disease_data/train/Herpes HPV and other STDs Photos', all_train_x)
        train_herpes_y = np.empty(len(train_herpes_x))
        train_herpes_y.fill(7)
        all_train_y.extend(train_herpes_y)

        train_light_diseases_x = self.load_images_from_folder('./skin_disease_data/train/Light Diseases and Disorders of Pigmentation', all_train_x)
        train_light_diseases_y = np.empty(len(train_light_diseases_x))
        train_light_diseases_y.fill(8)
        all_train_y.extend(train_light_diseases_y)

        train_lupus_x = self.load_images_from_folder('./skin_disease_data/train/Lupus and other Connective Tissue diseases', all_train_x)
        train_lupus_y = np.empty(len(train_lupus_x))
        train_lupus_y.fill(9)
        all_train_y.extend(train_lupus_y)

        train_melanoma_x = self.load_images_from_folder('./skin_disease_data/train/Melanoma Skin Cancer Nevi and Moles', all_train_x)
        train_melanoma_y = np.empty(len(train_melanoma_x))
        train_melanoma_y.fill(10)
        all_train_y.extend(train_melanoma_y)

        train_nail_fungus_x = self.load_images_from_folder('./skin_disease_data/train/Nail Fungus and other Nail Disease', all_train_x)
        train_nail_fungus_y = np.empty(len(train_nail_fungus_x))
        train_nail_fungus_y.fill(11)
        all_train_y.extend(train_nail_fungus_y)

        train_poison_ivy_x = self.load_images_from_folder('./skin_disease_data/train/Poison Ivy Photos and other Contact Dermatitis', all_train_x)
        train_poison_ivy_y = np.empty(len(train_poison_ivy_x))
        train_poison_ivy_y.fill(12)
        all_train_y.extend(train_poison_ivy_y)

        train_psoriasis_x = self.load_images_from_folder('./skin_disease_data/train/Psoriasis pictures Lichen Planus and related diseases', all_train_x)
        train_psoriasis_y = np.empty(len(train_psoriasis_x))
        train_psoriasis_y.fill(13)
        all_train_y.extend(train_psoriasis_y)

        train_scabies_lyme_x = self.load_images_from_folder('./skin_disease_data/train/Scabies Lyme Disease and other Infestations and Bites', all_train_x)
        train_scabies_lyme_y = np.empty(len(train_scabies_lyme_x))
        train_scabies_lyme_y.fill(14)
        all_train_y.extend(train_scabies_lyme_y)

        train_seborrheic_keratoses_x = self.load_images_from_folder('./skin_disease_data/train/Seborrheic Keratoses and other Benign Tumors', all_train_x)
        train_seborrheic_keratoses_y = np.empty(len(train_seborrheic_keratoses_x))
        train_seborrheic_keratoses_y.fill(15)
        all_train_y.extend(train_seborrheic_keratoses_y)

        train_systemic_disease_x = self.load_images_from_folder('./skin_disease_data/train/Systemic Disease', all_train_x)
        train_systemic_disease_y = np.empty(len(train_systemic_disease_x))
        train_systemic_disease_y.fill(16)
        all_train_y.extend(train_systemic_disease_y)

        train_tinea_ringworm_x = self.load_images_from_folder('./skin_disease_data/train/Tinea Ringworm Candidiasis and other Fungal Infections', all_train_x)
        train_tinea_ringworm_y = np.empty(len(train_tinea_ringworm_x))
        train_tinea_ringworm_y.fill(17)
        all_train_y.extend(train_tinea_ringworm_y)

        train_urticaria_hives_x = self.load_images_from_folder('./skin_disease_data/train/Urticaria Hives', all_train_x)
        train_urticaria_hives_y = np.empty(len(train_urticaria_hives_x))
        train_urticaria_hives_y.fill(18)
        all_train_y.extend(train_urticaria_hives_y)

        train_vascular_tumors_x = self.load_images_from_folder('./skin_disease_data/train/Vascular Tumors', all_train_x)
        train_vascular_tumors_y = np.empty(len(train_vascular_tumors_x))
        train_vascular_tumors_y.fill(19)
        all_train_y.extend(train_vascular_tumors_y)

        train_vasculitis_x = self.load_images_from_folder('./skin_disease_data/train/Vasculitis Photos', all_train_x)
        train_vasculitis_y = np.empty(len(train_vasculitis_x))
        train_vasculitis_y.fill(20)
        all_train_y.extend(train_vasculitis_y)

        train_warts_molluscum_x = self.load_images_from_folder('./skin_disease_data/train/Warts Molluscum and other Viral Infections', all_train_x)
        train_warts_molluscum_y = np.empty(len(train_warts_molluscum_x))
        train_warts_molluscum_y.fill(21)
        all_train_y.extend(train_warts_molluscum_y)


        test_x = np.array(all_test_x)
        test_y = np.array(all_test_y)
        train_x = np.array(all_train_x)
        train_y = np.array(all_train_y)

        prepped_train_x, prepped_train_y, prepped_test_x, prepped_test_y = self.prep_pixels(train_x, train_y, test_x, test_y)
        return prepped_train_x, prepped_train_y, prepped_test_x, prepped_test_y

def main():
    utils = MedicalDataUtils()
    utils.load_medical_data()

if __name__ == '__main__':
    main()