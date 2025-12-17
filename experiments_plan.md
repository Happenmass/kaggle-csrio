exp1

model inferenced feature: Dry_Clover_g, Dry_Dead_g, Dry_Green_g
rule calculated feature: Dry_Total_g = Dry_Clover_g + Dry_Dead_g + Dry_Green_g, GDM_g = Dry_Clover_g + Dry_Green_g

exp2

model inferenced feature: Dry_Clover_g, Dry_Total_g, Dry_Green_g
rule calculated feature: Dry_Dead_g = Dry_Total_g - (Dry_Clover_g + Dry_Green_g), GDM_g = Dry_Clover_g + Dry_Green_g

exp3

model inferenced feature: Dry_Total_g, Dry_Dead_g, Dry_Green_g
rule calculated feature: Dry_Clover_g = Dry_Total_g - (Dry_Dead_g + Dry_Green_g), GDM_g = Dry_Clover_g + Dry_Green_g

exp4

model inferenced feature: GDM_g, Dry_Clover_g, Dry_Dead_g
rule calculated feature: Dry_Clover_g = GDM_g - Dry_Green_g, Dry_Total_g = Dry_Clover_g + Dry_Dead_g + Dry_Green_g

exp5

model inferenced feature: GDM_g, Dry_Green_g, Dry_Dead_g
rule calculated feature: Dry_Clover_g = GDM_g - Dry_Green_g, Dry_Total_g = Dry_Clover_g + Dry_Dead_g + Dry_Green_g