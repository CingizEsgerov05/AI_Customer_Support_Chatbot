# backend.py - Professional Versiya
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import random
import os
import pickle
from difflib import SequenceMatcher
import re

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChatbotDataset:
    def __init__(self):
        self.intents = {
            "salamlama": {
                "patterns": [
                    "salam", "salam aleykum", "sabahınız xeyir", "axşamınız xeyir", 
                    "hey", "necəsən", "nə var nə yox", "salamlar", "alo", "hi",
                    "hello", "xoş gəlmisiniz", "salam olsun", "necəsiniz",
                    "gün aydın", "günün xeyir", "yaxşısınız", "hal necədir"
                ],
                "responses": [
                    "Salam! Sizə necə kömək edə bilərəm?",
                    "Xoş gəlmisiniz! Buyurun, sizi dinləyirəm.",
                    "Salam! Hansı məsələdə kömək edə bilərəm?",
                    "Hər vaxtınız xeyir! Nə ilə maraqlanırsınız?"
                ]
            },
            "sağollaşma": {
                "patterns": [
                    "sağ ol", "təşəkkürlər", "təşəkkür", "çox sağ ol", "minnətdaram",
                    "əlasınız", "başa düşdüm", "ok sağol", "təşəkkür edirəm",
                    "thank you", "thanks", "minnətdar qaldım", "çox gözəl",
                    "əla", "möhtəşəm", "super", "yaxşı kömək etdiniz"
                ],
                "responses": [
                    "Xahiş edirəm! Başqa bir şey lazımdır?",
                    "Buyurun, hər zaman xidmətinizdəyik!",
                    "Dəyməz! Sizə kömək etmək mənim vəzifəmdir.",
                    "Təşəkkür sizə! Başqa sualınız varsa, soruşun."
                ]
            },
            "vidalaşma": {
                "patterns": [
                    "əlvida", "hələlik", "görüşərik", "sağolun", "çıxış", "bay",
                    "gecəniz xeyrə", "görüşənədək", "çıxıram", "getməliyəm",
                    "getdim", "bəsdir", "kifayət", "sonra danışarıq"
                ],
                "responses": [
                    "Əlvida! Yenə gözləyirik.",
                    "Görüşənədək! Xoş vaxtlar!",
                    "Sağ olun, yaxşı yol!",
                    "Hələlik! Yenidən buyurun."
                ]
            },
            "mehsul_sorusu": {
                "patterns": [
                    "məhsul", "nə satırsınız", "katalog", "nələr var", "satış",
                    "hansı mallar var", "nəyiniz var", "çeşidlər", "brendlər",
                    "nə alım", "tövsiyə", "məhsul göstər", "seçim", "məhsullar",
                    "ən yaxşı", "hansını alım", "populyar", "ən çox satılan",
                    "yeni gələnlər", "kataloq göstər"
                ],
                "responses": [
                    "Bizdə Elektronika, geyim, ev əşyaları və aksesuarlar var. Hansı kateqoriya maraqlandırır?",
                    "Geniş çeşidimiz var: Telefonlar, noutbuklar, geyim, kosmetika və daha çox! Kataloqumuza saytımızdan baxa bilərsiniz.",
                    "Əsas kateqoriyalar: 📱 Elektronika, 👔 Geyim, 🏠 Ev əşyaları, 🎮 Oyun aksessuarları. Hansına baxmaq istəyirsiniz?"
                ]
            },
            "qiymet_sorusu": {
                "patterns": [
                    "qiymət", "neçəyədir", "qiyməti", "nə qədərdir", "ödəniş",
                    "qiyməti deyin", "neçiyə", "qaçadır", "baha", "ucuz", "qiymet",
                    "pul", "məbləğ", "dəyəri", "maya dəyəri", "ən ucuz",
                    "ən baha", "orta qiymət", "qiymət aralığı"
                ],
                "responses": [
                    "Qiymətlər məhsula görə dəyişir. Hansı məhsulla maraqlanırsınız?",
                    "Zəhmət olmasa məhsulun adını dəqiq yazın, qiymətini yoxlayım.",
                    "Ən ucuz məhsullarımız 10 AZN-dən, premium kateqoriya 500+ AZN-dən başlayır. Nəyə baxırsınız?"
                ]
            },
            "catdirilma": {
                "patterns": [
                    "çatdırılma", "kuryer", "gətirilmə", "nə vaxt gəlir", "karqo",
                    "rayonlara çatdırılma", "sifariş nə vaxt çatar", "evə çatdırılma",
                    "çatdırma müddəti", "çatdırılır", "göndərmə", "çatdırma haqqı",
                    "pulsuz çatdırılma", "express", "tez çatdırılma", "ləng çatdırılma"
                ],
                "responses": [
                    "Bakı daxili çatdırılma 1 iş günü, rayonlara 2-3 iş günü çəkir. 🚚",
                    "50 AZN və üzəri sifarişlərdə çatdırılma pulsuzdur! Qapıya qədər gətirilir.",
                    "Express çatdırılma: 4 saat ərzində (+5 AZN). Standart: 1-2 gün (pulsuz 50+ AZN)."
                ]
            },
            "destek": {
                "patterns": [
                    "dəstək", "kömək", "problem", "xəta", "işləmir", "operator",
                    "canlı operator", "insanla danışmaq", "şikayət", "səhv",
                    "narazılıq", "məsələ", "qırıldı", "sıradan çıxdı", "admin",
                    "müraciət", "düzəlt", "help", "support", "texniki yardım"
                ],
                "responses": [
                    "Probleminizi ətraflı izah edin, həll etməyə çalışım. 🛠️",
                    "Texniki dəstək komandamız sizinlədir! Nə probleminiz var?",
                    "Canlı operatorla əlaqə: +994 XX XXX XX XX və ya info@example.az"
                ]
            },
            "unvan": {
                "patterns": [
                    "ünvan", "harda yerləşirsiniz", "yeriniz", "ofisiniz hardadır",
                    "hansı metro", "lokasiya", "xəritə", "filial", "ünvanınız",
                    "harada", "məkan", "location", "address", "mağaza",
                    "showroom", "ofis", "şöbə"
                ],
                "responses": [
                    "📍 Baş ofis: Bakı şəhəri, 28 May metrosu yaxınlığı, Nizami küç. 123",
                    "Filiallarımız: Gənclik Mall, Park Bulvar, 28 Mall. Xəritə: [link]",
                    "Anbar: Nərimanov rayonu. Onlayn sifariş vermək üçün saytımızdan istifadə edin."
                ]
            },
            "is_saatlari": {
                "patterns": [
                    "saat neçədə", "iş saatları", "nə vaxt açılır", "açıqsınız",
                    "iş vaxtı", "günorta fasiləsi", "həftə sonu işləyirsiniz",
                    "neçədən neçəyə", "bazar günü", "bağlanma vaxtı", "açılma saatı",
                    "iş günləri", "fasiləsiz"
                ],
                "responses": [
                    "⏰ Həftə içi: 09:00-18:00, Şənbə: 10:00-16:00. Bazar günü istirahətdir.",
                    "Onlayn xidmət 24/7 fəaliyyətdədir! Mağaza: 09:00-18:00",
                    "Call mərkəz: Həftə içi 09:00-20:00, həftə sonu 10:00-18:00"
                ]
            },
            "odeme_usullari": {
                "patterns": [
                    "ödəniş üsulları", "kart keçir", "nağd", "kredit", "taksit",
                    "birbank", "kapital", "terminal", "necə ödəyə bilərəm",
                    "ödəmə formaları", "bank kartı", "online ödəniş", "pos terminal",
                    "mastercard", "visa", "təqsit", "faizsiz", "ödəniş seçimləri"
                ],
                "responses": [
                    "💳 Ödəniş üsulları: Nağd, Bank kartı (Visa/Master), Birbank, Kapital.",
                    "Taksit: 3-6-12 ay faizsiz (Birbank, TamKart, BirKart ilə).",
                    "Qapıda ödəniş və ya onlayn ödəmə - seçim sizindir!"
                ]
            },
            "qaytarma": {
                "patterns": [
                    "qaytarmaq", "dəyişdirmək", "geri qaytarma", "bəyənmədim",
                    "ölçü səhvdir", "zəmanət", "iadə", "dəyişdirə bilərəm",
                    "problem var", "xarab gəldi", "defekt", "sınıq", "zədəli",
                    "uyğun gəlmədi", "rəng fərqli", "nömrə kiçikdir"
                ],
                "responses": [
                    "✅ 14 gün ərzində qəbz ilə geri qayta və ya dəyişdirə bilərsiniz (zədəsiz).",
                    "Ölçü problemi? Heç problem deyil - dəyişdiririk! Sadəcə qəbzi gətirin.",
                    "Zəmanətli məhsullar: Texniki xəta halında təmir və ya dəyişiklik."
                ]
            },
            "endirim": {
                "patterns": [
                    "endirim", "kampaniya", "aksiya", "ucuzluq", "sale", "güzəşt",
                    "promokod", "kod", "kupon", "kompaniya", "təklif", "bonus",
                    "endirimlər", "black friday", "sezon endirimi", "yeni il endirimi",
                    "promo", "discount", "offer"
                ],
                "responses": [
                    "🔥 Aktiv kompaniya: Seçilmiş elektronikada 20% endirim! Kod: TECH20",
                    "💰 Promokodlar: Saytın 'Endirimlər' bölməsinə baxın və ya email-lə qeydiyyat olun.",
                    "Təəssüf ki, hazırda aktiv kompaniya yoxdur. Gözləyin, tezliklə yeniləri olacaq!"
                ]
            }
        }

    def get_training_data(self):
        data = []
        for intent, content in self.intents.items():
            for pattern in content['patterns']:
                data.append({
                    'text': pattern.lower(),
                    'intent': intent,
                    'response': random.choice(content['responses'])
                })
        return data

    def augment_data(self):
        """Məlumatları süni şəkildə artırır (data augmentation)"""
        augmented = []
        synonyms = {
            'salam': ['salam', 'salamlar', 'hey', 'hi'],
            'məhsul': ['məhsul', 'mal', 'product'],
            'qiymət': ['qiymət', 'qiymet', 'pul', 'məbləğ'],
            'çatdırılma': ['çatdırılma', 'gətirilmə', 'kuryer']
        }
        
        for item in self.get_training_data():
            augmented.append(item)
            text = item['text']
            for word, syns in synonyms.items():
                if word in text:
                    for syn in syns:
                        new_text = text.replace(word, syn)
                        if new_text != text:
                            augmented.append({
                                'text': new_text,
                                'intent': item['intent'],
                                'response': item['response']
                            })
        return augmented


class BERTChatbot(nn.Module):
    def __init__(self, num_intents, dropout=0.3):
        super(BERTChatbot, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        # Daha güclü classifier
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_intents)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        x = self.dropout(pooled)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class ChatbotInterface:
    def __init__(self, model, tokenizer, intent_labels, dataset_obj):
        self.model = model
        self.tokenizer = tokenizer
        self.intent_labels = intent_labels
        self.intents_data = dataset_obj.intents
        self.device = DEVICE
        self.model.to(self.device)
        self.model.eval()
    
    def clean_text(self, text):
        """Mətni təmizləyir və normallaşdırır"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Durğu işarələrini sil
        return text
    
    def fuzzy_match(self, text, patterns):
        """Yaxın oxşarlıqları tapır (typo tolerance)"""
        best_ratio = 0
        for pattern in patterns:
            ratio = SequenceMatcher(None, text, pattern).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
        return best_ratio
    
    def keyword_scoring(self, text):
        """Açar söz əsaslı xal hesablama"""
        intent_scores = {intent: 0 for intent in self.intents_data}
        
        for intent, data in self.intents_data.items():
            for pattern in data['patterns']:
                # Tam uyğunluq
                if pattern in text:
                    intent_scores[intent] += len(pattern.split()) * 2
                # Fuzzy match (80%+ oxşarlıq)
                elif self.fuzzy_match(text, [pattern]) > 0.8:
                    intent_scores[intent] += len(pattern.split())
        
        max_score = max(intent_scores.values())
        best_intent = max(intent_scores, key=intent_scores.get) if max_score > 0 else None
        
        return best_intent, max_score
    
    def get_response(self, text):
        clean_text = self.clean_text(text)
        
        # Boş input yoxlama
        if not clean_text:
            return "Zəhmət olmasa sualınızı yazın."
        
        # 1. Keyword-based matching (sürətli və dəqiq)
        best_intent, score = self.keyword_scoring(clean_text)
        
        if best_intent and score >= 2:  # Güclü uyğunluq
            return random.choice(self.intents_data[best_intent]['responses'])
        
        # 2. BERT model (daha mürəkkəb hallar üçün)
        try:
            encoding = self.tokenizer.encode_plus(
                clean_text,
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            conf_value = confidence.item()
            
            # Dinamik threshold (keyword score-a görə)
            threshold = 0.6 if score > 0 else 0.7
            
            if conf_value < threshold:
                return (
                    "Üzr istəyirəm, sualınızı tam başa düşə bilmədim. 🤔\n\n"
                    "Belə suallar verə bilərsiniz:\n"
                    "• Məhsullarınız haqqında\n"
                    "• Qiymətlər və endirimlər\n"
                    "• Çatdırılma və ödəniş\n"
                    "• Qaytarma şərtləri"
                )
            
            intent = self.intent_labels[predicted.item()]
            response = random.choice(self.intents_data[intent]['responses'])
            
            # Aşağı confidence-də xəbərdarlıq əlavə et
            if conf_value < 0.75:
                response += "\n\n(Əgər cavab tam dəqiq deyilsə, sualı başqa cür yazmağa çalışın)"
            
            return response
            
        except Exception as e:
            print(f"Error: {e}")
            return "Sistemdə texniki xəta baş verdi. Zəhmət olmasa bir az sonra yenidən cəhd edin."


def load_system():
    if not os.path.exists('best_chatbot_model.pth'):
        raise FileNotFoundError(
            "Model tapılmadı! Zəhmət olmasa əvvəlcə 'python train.py' əmrini işlədin."
        )

    with open('chatbot_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BERTChatbot(num_intents=len(metadata['label_to_intent']))
    model.load_state_dict(torch.load('best_chatbot_model.pth', map_location=DEVICE))
    
    return ChatbotInterface(
        model, 
        tokenizer, 
        metadata['label_to_intent'], 
        metadata['dataset_obj']
    )