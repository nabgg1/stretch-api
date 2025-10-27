import librosa
import soundfile as sf
import pyrubberband as rb
import numpy as np

# La fonction accepte maintenant un nouveau paramètre : duree_cible_str
def stretch_audio(chemin_entree, chemin_sortie, duree_cible_str):
    
    try:
        # --- ÉTAPE 1 : CHARGER LE FICHIER ---
        print(f"Chargement du fichier : {chemin_entree}")
        y, sr = librosa.load(chemin_entree, sr=None, mono=False)
        y = np.clip(y, -1.0, 1.0) # Sécurité anti-clipping

        # --- ÉTAPE 2 : CALCULER LA DURÉE ORIGINALE ---
        duree_originale = librosa.get_duration(y=y, sr=sr)
        
        # On convertit notre durée cible (envoyée par Bubble) en nombre
        duree_cible = float(duree_cible_str)
        
        print(f"Durée originale : {duree_originale:.2f}s")
        print(f"Durée cible demandée : {duree_cible:.2f}s")

        # --- ÉTAPE 3 : NOUVELLE CONDITION DE TOLÉRANCE ---
        # On calcule la différence absolue entre l'original et la cible
        difference = abs(duree_originale - duree_cible)
        print(f"Différence : {difference:.2f}s")
        
        # Si la différence est dans la tolérance (<= 0.6s), on ne fait rien.
        if difference <= 0.6:
            print("La durée est déjà dans la tolérance (+/- 0.6s). Aucun stretching n'est appliqué.")
            # On renvoie le chemin du fichier original non modifié
            return chemin_entree
        
        # --- ÉTAPE 4 : STRETCHING NÉCESSAIRE ---
        print("Stretching en cours vers la cible...")
        
        # On calcule le "rate" pour atteindre la durée cible exacte
        rate = duree_originale / duree_cible
        print(f"Rate pour RubberBand : {rate:.2f}")

        # Conversion en 32-bit pour la qualité
        y_int32 = (y.T * 2147483647).astype(np.int32)
        
        # Appliquer le time-stretch avec nos meilleurs réglages
        y_modifie = rb.time_stretch(y_int32, sr, rate=rate, rbargs={'--formant': ''})

        # --- ÉTAPE 5 : SAUVEGARDER LE NOUVEAU FICHIER ---
        sf.write(chemin_sortie, y_modifie, sr, subtype='PCM_32')
        
        print(f"Terminé ! Fichier modifié sauvegardé sous : {chemin_sortie}")
        return chemin_sortie

    except Exception as e:
        print(f"Une erreur est survenue lors du stretching : {e}")
        return None
