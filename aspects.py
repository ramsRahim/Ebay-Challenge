aspect_names = {
    "No Tag": "No Tag",
    "Obscure": "Obscure",
    "Abteilung": "Department",
    "Aktivität": "Performance/Activity",
    "Akzente": "Accents",
    "Anlass": "Occasion",
    "Besonderheiten": "Features",
    "Charakter": "Character",
    "Charakter Familie": "Character Family",
    "Dämpfungsgrad": "Cushioning Level",
    "Erscheinungsjahr": "Release Year",
    "EU-Schuhgröße": "EU Shoe Size",
    "Farbe": "Color",
    "Futtermaterial": "Lining Material",
    "Gewebeart": "Fabric Type",
    "Herstellernummer": "Style Code",
    "Herstellungsland und -region": "Country/Region of Manufacture",
    "Innensohlenmaterial": "Insole Material",
    "Jahreszeit": "Season",
    "Laufsohlenmaterial": "Outsole Material",
    "Marke": "Brand",
    "Maßeinheit": "Unit of Measure",
    "Modell": "Model",
    "Muster": "Pattern",
    "Obermaterial": "Upper Material",
    "Produktart": "Type",
    "Produktlinie": "Product Line",
    "Schuhschaft-Typ": "Shoe Shaft Style",
    "Schuhweite": "Shoe Width",
    "Stil": "Style",
    "Stollentyp": "Cleat Type",
    "Thema": "Theme",
    "UK-Schuhgröße": "UK Shoe Size",
    "US-Schuhgröße": "US Shoe Size",
    "Verschluss": "Closure",
    "Zwischensohlen-Typ": "Midsole Type"
}


name_to_id = {name: id for id, name in enumerate(aspect_names)}

total_ids = len(name_to_id)
keys = list(name_to_id.keys())
for name in keys:
    if name not in ['No Tag', 'Obscure']:
        name_to_id[name + '-I'] = total_ids
        total_ids += 1

print(name_to_id)

id_to_name = {id: name for name, id in name_to_id.items()}
    