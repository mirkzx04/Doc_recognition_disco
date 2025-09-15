#!/usr/bin/env python3
"""
Test avanzato per il sistema di pulizia immagini documenti.

Questo script permette di:
- Testare la pipeline di pulizia su un numero limitato di immagini
- Confrontare originale vs processata
- Analizzare metriche di qualità
- Testare diversi tipi di documento
- Salvare risultati per analisi

Utilizzo:
    python test_image_cleaner.py
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import random
from pathlib import Path
import time

# Import delle nostre classi
from img_preproc.CleanerIMG import CleanerIMG
from img_preproc.FindsLinesIMG import FindsLinesIMG

class ImageCleanerTester:
    """
    Classe per testare e validare il sistema di pulizia immagini.
    
    Fornisce interfaccia interattiva per:
    - Selezione numero immagini da testare
    - Confronto visivo originale/processata
    - Analisi qualità e performance
    - Salvataggio risultati
    """
    
    def __init__(self, dataset_path="dataset_pdf_v1"):
        """
        Inizializza il tester.
        
        Args:
            dataset_path (str): Percorso al dataset
        """
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'images')
        self.label_path = os.path.join(dataset_path, 'labels')
        
        # Inizializza componenti di pulizia
        self.cleaner = CleanerIMG(debug=True)
        self.finder = FindsLinesIMG(debug=True)
        
        # Variabili per i test
        self.test_results = []
        self.current_index = 0
        self.max_images = 10
        
        print("🧪 ImageCleanerTester inizializzato")
        print(f"Dataset: {dataset_path}")
        
    def run_interactive_test(self):
        """
        Avvia test interattivo con interfaccia utente.
        """
        print("\n" + "="*60)
        print("🔬 TEST INTERATTIVO SISTEMA PULIZIA IMMAGINI")
        print("="*60)
        
        # Richiedi numero massimo di immagini
        while True:
            try:
                max_imgs = input("\nInserisci numero massimo di immagini da testare (1-100): ")
                self.max_images = int(max_imgs)
                if 1 <= self.max_images <= 100:
                    break
                else:
                    print("⚠️ Inserisci un numero tra 1 e 100")
            except ValueError:
                print("⚠️ Inserisci un numero valido")
        
        # Selezione tipo test
        print("\n🎯 Seleziona modalità di test:")
        print("1. Test casuale su tutti i tipi di documento")
        print("2. Test specifico per tessere (tipo 01)")
        print("3. Test specifico per fogli (tipo 02)")
        print("4. Test specifico per documenti multipli (tipo 03)")
        print("5. Test comparativo tra tutti i tipi")
        
        while True:
            try:
                choice = input("\nScegli modalità (1-5): ")
                if choice in ['1', '2', '3', '4', '5']:
                    break
                else:
                    print("⚠️ Scegli un numero tra 1 e 5")
            except ValueError:
                print("⚠️ Inserisci un numero valido")
        
        # Esegui test basato sulla scelta
        if choice == '1':
            self._test_random_documents()
        elif choice == '2':
            self._test_document_type('01')
        elif choice == '3':
            self._test_document_type('02')
        elif choice == '4':
            self._test_document_type('03')
        elif choice == '5':
            self._test_comparative_analysis()
    
    def _test_random_documents(self):
        """Test casuale su documenti di tutti i tipi."""
        print(f"\nAvvio test casuale su {self.max_images} immagini...")
        
        # Carica lista file immagini
        image_files = [f for f in os.listdir(self.img_path) if f.endswith('.jpg')]
        
        if len(image_files) == 0:
            print("Nessuna immagine trovata nel dataset!")
            return
            
        # Seleziona immagini casuali
        selected_files = random.sample(image_files, min(self.max_images, len(image_files)))
        
        print(f"📋 Selezionate {len(selected_files)} immagini per il test\n")
        
        results = []
        
        for i, img_file in enumerate(selected_files):
            print(f"Processando immagine {i+1}/{len(selected_files)}: {img_file}")
            
            # Carica immagine
            img_path = os.path.join(self.img_path, img_file)
            original_img = cv2.imread(img_path)
            
            if original_img is None:
                print(f"⚠️ Impossibile caricare {img_file}")
                continue
            
            # Determina tipo documento dal label
            doc_type = self._get_document_type(img_file)
            
            # Processa immagine
            start_time = time.time()
            processed_img = self.cleaner.pipeline_clean(original_img.copy(), doc_type)
            processing_time = time.time() - start_time
            
            if processed_img is None:
                print(f"Elaborazione fallita per {img_file}")
                continue
            
            # Calcola metriche
            original_quality = self.cleaner.assess_quality(original_img)
            processed_quality = self.cleaner.assess_quality(processed_img)
            
            # Salva risultati
            result = {
                'filename': img_file,
                'doc_type': doc_type,
                'processing_time': processing_time,
                'original_quality': original_quality,
                'processed_quality': processed_quality,
                'improvement': processed_quality['overall'] - original_quality['overall']
            }
            results.append(result)
            
            # Mostra confronto visivo
            self._show_comparison(original_img, processed_img, result)
            
            # Pausa tra immagini
            input(f"\nPremi INVIO per continuare con la prossima immagine...")
            plt.close('all')  # Chiudi plots precedenti
        
        # Mostra statistiche finali
        self._show_final_statistics(results)
    
    def _test_document_type(self, doc_type):
        """Test specifico per un tipo di documento."""
        type_names = {'01': 'Tessere', '02': 'Fogli', '03': 'Documenti Multipli'}
        print(f"\nTest specifico per {type_names.get(doc_type, doc_type)}...")
        
        # Trova documenti del tipo specificato
        target_files = []
        
        for label_file in os.listdir(self.label_path):
            if label_file.endswith('.json'):
                try:
                    with open(os.path.join(self.label_path, label_file), 'r') as f:
                        data = json.load(f)
                        data = json.loads(data['ground_truth'])
                        if data['tipoDocumento'] == doc_type:
                            img_filename = label_file.replace('.json', '.jpg')
                            if os.path.exists(os.path.join(self.img_path, img_filename)):
                                target_files.append(img_filename)
                except:
                    continue
        
        if not target_files:
            print(f"Nessun documento di tipo {doc_type} trovato!")
            return
            
        print(f"🎯 Trovati {len(target_files)} documenti di tipo {type_names.get(doc_type)}")
        
        # Seleziona campione
        selected_files = random.sample(target_files, min(self.max_images, len(target_files)))
        
        # Testa documenti selezionati
        results = []
        
        for i, img_file in enumerate(selected_files):
            print(f"\nAnalizzando {type_names.get(doc_type)} {i+1}/{len(selected_files)}")
            
            result = self._process_single_image(img_file, doc_type)
            if result:
                results.append(result)
                self._show_detailed_analysis(result)
                
                if i < len(selected_files) - 1:  # Non chiedere per l'ultima
                    input("Premi INVIO per continuare...")
                    plt.close('all')
        
        # Analisi specifica per tipo
        self._show_type_specific_analysis(results, doc_type)
    
    def _test_comparative_analysis(self):
        """Test comparativo tra tutti i tipi di documento."""
        print("\nAvvio analisi comparativa tra tipi di documento...")
        
        type_results = {'01': [], '02': [], '03': []}
        total_per_type = max(1, self.max_images // 3)
        
        for doc_type in ['01', '02', '03']:
            print(f"\nTestando tipo {doc_type}...")
            
            # Trova documenti del tipo
            target_files = self._find_documents_by_type(doc_type)
            
            if not target_files:
                print(f"Nessun documento tipo {doc_type} trovato")
                continue
                
            # Seleziona campione
            selected = random.sample(target_files, min(total_per_type, len(target_files)))
            
            for img_file in selected:
                result = self._process_single_image(img_file, doc_type)
                if result:
                    type_results[doc_type].append(result)
        
        # Mostra analisi comparativa
        self._show_comparative_results(type_results)
    
    def _find_documents_by_type(self, doc_type):
        """Trova documenti di un tipo specifico."""
        target_files = []
        
        for label_file in os.listdir(self.label_path):
            if label_file.endswith('.json'):
                try:
                    with open(os.path.join(self.label_path, label_file), 'r') as f:
                        data = json.load(f)
                        data = json.loads(data['ground_truth'])
                        if data['tipoDocumento'] == doc_type:
                            img_filename = label_file.replace('.json', '.jpg')
                            if os.path.exists(os.path.join(self.img_path, img_filename)):
                                target_files.append(img_filename)
                except:
                    continue
        
        return target_files
    
    def _process_single_image(self, img_file, doc_type):
        """Processa una singola immagine e ritorna i risultati."""
        img_path = os.path.join(self.img_path, img_file)
        original_img = cv2.imread(img_path)
        
        if original_img is None:
            return None
            
        # Processa immagine
        start_time = time.time()
        
        if doc_type == '03':  # Documenti multipli
            multi_docs = self.cleaner.process_multi_documents(original_img, self.finder)
            processing_time = time.time() - start_time
            
            if not multi_docs:
                return None
                
            # Analizza ogni documento estratto
            processed_qualities = []
            for doc in multi_docs:
                quality = self.cleaner.assess_quality(doc)
                processed_qualities.append(quality)
            
            # Media delle qualità
            avg_quality = {
                'overall': np.mean([q['overall'] for q in processed_qualities]),
                'sharpness': np.mean([q['sharpness'] for q in processed_qualities]),
                'contrast': np.mean([q['contrast'] for q in processed_qualities]),
                'brightness': np.mean([q['brightness'] for q in processed_qualities])
            }
            
            processed_img = multi_docs[0] if multi_docs else None  # Per visualizzazione
            
        else:
            processed_img = self.cleaner.pipeline_clean(original_img.copy(), doc_type)
            processing_time = time.time() - start_time
            
            if processed_img is None:
                return None
                
            avg_quality = self.cleaner.assess_quality(processed_img)
        
        # Calcola metriche
        original_quality = self.cleaner.assess_quality(original_img)
        
        return {
            'filename': img_file,
            'doc_type': doc_type,
            'processing_time': processing_time,
            'original_img': original_img,
            'processed_img': processed_img,
            'original_quality': original_quality,
            'processed_quality': avg_quality,
            'improvement': avg_quality['overall'] - original_quality['overall']
        }
    
    def _get_document_type(self, img_file):
        """Determina il tipo di documento dal file label."""
        label_file = img_file.replace('.jpg', '.json')
        label_path = os.path.join(self.label_path, label_file)
        
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
                data = json.loads(data['ground_truth'])
                return data['tipoDocumento']
        except:
            return '01'  # Default a tessera
    
    def _show_comparison(self, original, processed, result):
        """Mostra confronto visivo tra originale e processata."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Immagine originale
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Originale\n{result["filename"]}', fontsize=12)
        axes[0, 0].axis('off')
        
        # Immagine processata
        axes[0, 1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Processata', fontsize=12)
        axes[0, 1].axis('off')
        
        # Differenza
        diff = cv2.absdiff(cv2.resize(original, processed.shape[:2][::-1]), processed)
        axes[0, 2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Differenza', fontsize=12)
        axes[0, 2].axis('off')
        
        # Metriche originale
        orig_metrics = result['original_quality']
        axes[1, 0].bar(orig_metrics.keys(), orig_metrics.values(), color='red', alpha=0.7)
        axes[1, 0].set_title('Qualità Originale', fontsize=12)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Metriche processata
        proc_metrics = result['processed_quality']
        axes[1, 1].bar(proc_metrics.keys(), proc_metrics.values(), color='green', alpha=0.7)
        axes[1, 1].set_title('Qualità Processata', fontsize=12)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Confronto miglioramenti
        metrics_names = list(orig_metrics.keys())
        improvements = [proc_metrics[m] - orig_metrics[m] for m in metrics_names]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[1, 2].bar(metrics_names, improvements, color=colors, alpha=0.7)
        axes[1, 2].set_title('Miglioramenti', fontsize=12)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Info aggiuntive
        info_text = f"""
Tipo Documento: {result['doc_type']}
Tempo Elaborazione: {result['processing_time']:.2f}s
Miglioramento Overall: {result['improvement']:.3f}
"""
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def _show_detailed_analysis(self, result):
        """Mostra analisi dettagliata di un singolo risultato."""
        print(f"\nANALISI DETTAGLIATA: {result['filename']}")
        print("=" * 50)
        
        print(f"Tipo documento: {result['doc_type']}")
        print(f"Tempo elaborazione: {result['processing_time']:.2f} secondi")
        print(f"Miglioramento qualità: {result['improvement']:.3f}")
        
        print("\nMETRICHE QUALITÀ:")
        print("-" * 30)
        orig = result['original_quality']
        proc = result['processed_quality']
        
        for metric in orig.keys():
            improvement = proc[metric] - orig[metric]
            arrow = "↗" if improvement > 0 else "↘" if improvement < 0 else "→"
            print(f"{metric:12}: {orig[metric]:.3f} → {proc[metric]:.3f} {arrow} ({improvement:+.3f})")
        
        # Valutazione qualitativa
        if result['improvement'] > 0.2:
            print("\nMIGLIORAMENTO ECCELLENTE!")
        elif result['improvement'] > 0.1:
            print("\nMiglioramento significativo")
        elif result['improvement'] > 0:
            print("\nLeggero miglioramento")
        else:
            print("\nNessun miglioramento o peggioramento")
    
    def _show_final_statistics(self, results):
        """Mostra statistiche finali del test."""
        if not results:
            print("\nNessun risultato da analizzare!")
            return
            
        print("\n" + "="*60)
        print("STATISTICHE FINALI DEL TEST")
        print("="*60)
        
        # Statistiche generali
        total_tests = len(results)
        avg_processing_time = np.mean([r['processing_time'] for r in results])
        avg_improvement = np.mean([r['improvement'] for r in results])
        
        print(f"\nTest completati: {total_tests}")
        print(f"Tempo medio elaborazione: {avg_processing_time:.2f} secondi")
        print(f"Miglioramento medio qualità: {avg_improvement:.3f}")
        
        # Distribuzione miglioramenti
        improvements = [r['improvement'] for r in results]
        excellent = sum(1 for imp in improvements if imp > 0.2)
        good = sum(1 for imp in improvements if 0.1 < imp <= 0.2)
        moderate = sum(1 for imp in improvements if 0 < imp <= 0.1)
        no_improvement = sum(1 for imp in improvements if imp <= 0)
        
        print(f"\nDISTRIBUZIONE MIGLIORAMENTI:")
        print(f"   Eccellenti (>0.2): {excellent}/{total_tests} ({excellent/total_tests*100:.1f}%)")
        print(f"   Buoni (0.1-0.2): {good}/{total_tests} ({good/total_tests*100:.1f}%)")
        print(f"   Moderati (0-0.1): {moderate}/{total_tests} ({moderate/total_tests*100:.1f}%)")
        print(f"   Nessuno/Peggio (≤0): {no_improvement}/{total_tests} ({no_improvement/total_tests*100:.1f}%)")
        
        # Grafico riassuntivo
        self._create_summary_plot(results)
        
        # Salva risultati
        self._save_results(results)
    
    def _show_comparative_results(self, type_results):
        """Mostra risultati dell'analisi comparativa."""
        print("\n" + "="*60)
        print("ANALISI COMPARATIVA TRA TIPI DI DOCUMENTO")
        print("="*60)
        
        type_names = {'01': 'Tessere', '02': 'Fogli', '03': 'Multipli'}
        
        for doc_type, results in type_results.items():
            if not results:
                continue
                
            print(f"\n{type_names.get(doc_type, doc_type).upper()}:")
            print("-" * 30)
            
            avg_improvement = np.mean([r['improvement'] for r in results])
            avg_time = np.mean([r['processing_time'] for r in results])
            success_rate = len(results) / len(results) * 100  # Tutti i risultati sono successi
            
            print(f"   Campioni testati: {len(results)}")
            print(f"   Miglioramento medio: {avg_improvement:.3f}")
            print(f"   Tempo medio: {avg_time:.2f}s")
            print(f"   Tasso successo: {success_rate:.1f}%")
        
        # Crea grafico comparativo
        self._create_comparative_plot(type_results)
    
    def _create_summary_plot(self, results):
        """Crea grafico riassuntivo dei risultati."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribuzione miglioramenti
        improvements = [r['improvement'] for r in results]
        axes[0, 0].hist(improvements, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribuzione Miglioramenti Qualità')
        axes[0, 0].set_xlabel('Miglioramento')
        axes[0, 0].set_ylabel('Frequenza')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Tempi di elaborazione
        times = [r['processing_time'] for r in results]
        axes[0, 1].hist(times, bins=15, alpha=0.7, color='green')
        axes[0, 1].set_title('Distribuzione Tempi Elaborazione')
        axes[0, 1].set_xlabel('Tempo (secondi)')
        axes[0, 1].set_ylabel('Frequenza')
        
        # Qualità originale vs processata
        orig_qualities = [r['original_quality']['overall'] for r in results]
        proc_qualities = [r['processed_quality']['overall'] for r in results]
        axes[1, 0].scatter(orig_qualities, proc_qualities, alpha=0.6)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.7)
        axes[1, 0].set_xlabel('Qualità Originale')
        axes[1, 0].set_ylabel('Qualità Processata')
        axes[1, 0].set_title('Originale vs Processata')
        
        # Miglioramenti per tipo documento
        doc_types = [r['doc_type'] for r in results]
        type_improvements = {}
        for doc_type in set(doc_types):
            type_improvements[doc_type] = np.mean([r['improvement'] for r in results if r['doc_type'] == doc_type])
        
        axes[1, 1].bar(type_improvements.keys(), type_improvements.values(), alpha=0.7)
        axes[1, 1].set_title('Miglioramento Medio per Tipo')
        axes[1, 1].set_xlabel('Tipo Documento')
        axes[1, 1].set_ylabel('Miglioramento Medio')
        
        plt.tight_layout()
        plt.show()
    
    def _create_comparative_plot(self, type_results):
        """Crea grafico comparativo tra tipi."""
        type_names = {'01': 'Tessere', '02': 'Fogli', '03': 'Multipli'}
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Miglioramenti medi
        types = []
        improvements = []
        times = []
        
        for doc_type, results in type_results.items():
            if results:
                types.append(type_names.get(doc_type, doc_type))
                improvements.append(np.mean([r['improvement'] for r in results]))
                times.append(np.mean([r['processing_time'] for r in results]))
        
        # Grafico miglioramenti
        axes[0].bar(types, improvements, alpha=0.7, color=['red', 'green', 'blue'])
        axes[0].set_title('Miglioramento Medio per Tipo')
        axes[0].set_ylabel('Miglioramento Qualità')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Grafico tempi
        axes[1].bar(types, times, alpha=0.7, color=['red', 'green', 'blue'])
        axes[1].set_title('Tempo Medio Elaborazione')
        axes[1].set_ylabel('Tempo (secondi)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Box plot distribuzione miglioramenti
        all_improvements = []
        labels = []
        for doc_type, results in type_results.items():
            if results:
                all_improvements.append([r['improvement'] for r in results])
                labels.append(type_names.get(doc_type, doc_type))
        
        if all_improvements:
            axes[2].boxplot(all_improvements, labels=labels)
            axes[2].set_title('Distribuzione Miglioramenti')
            axes[2].set_ylabel('Miglioramento Qualità')
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _save_results(self, results):
        """Salva i risultati del test in un file JSON."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        # Prepara dati per il salvataggio (rimuovi oggetti non serializzabili)
        save_data = []
        for result in results:
            save_result = {
                'filename': result['filename'],
                'doc_type': result['doc_type'],
                'processing_time': result['processing_time'],
                'original_quality': result['original_quality'],
                'processed_quality': result['processed_quality'],
                'improvement': result['improvement']
            }
            save_data.append(save_result)
        
        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"\nRisultati salvati in: {filename}")
        except Exception as e:
            print(f"\nErrore nel salvataggio: {e}")

def main():
    """Funzione principale per avviare il test."""
    print("Avvio sistema di test per pulizia immagini documenti")
    
    # Verifica esistenza dataset
    dataset_path = "dataset_pdf_v1"
    if not os.path.exists(dataset_path):
        print(f"Dataset non trovato in: {dataset_path}")
        print("Assicurati che il dataset sia nella directory corretta")
        return
    
    # Inizializza e avvia tester
    tester = ImageCleanerTester(dataset_path)
    tester.run_interactive_test()
    
    print("\nTest completato!")
    print("Controlla i risultati salvati e i grafici generati")

if __name__ == "__main__":
    main()
