def postprocess(gnc_results: dict, output_dir: str):
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gnc_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(gnc_results, f, indent=4)
    
    print(f"GNC results have been saved to {output_path}")