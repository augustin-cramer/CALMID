from river.datasets.synth import ConceptDriftStream, RandomRBFDrift, RandomRBF, RandomTree
import pandas as pd
from tqdm import tqdm

def create_dataset(n_iterations_1, n_iterations_2, n_iterations_3, generator_1, generator_2, generator_3):

    ## This function can approximate only synthetic datasets with sudden drift

    df1 = pd.DataFrame({'0' : [], '1' : [], '2' : [], '3' : [], '4' : [], '5' : [], '6' : [], '7' : [], '8' : [], '9' : [], '10' : [], '11' : [], '12' : [], '13' : [], '14' : [], '15' : [], '16' : [], '17' : [], '18' : [], '19' : [], '20' : [], 'label' : []})
    df2 = pd.DataFrame({'0' : [], '1' : [], '2' : [], '3' : [], '4' : [], '5' : [], '6' : [], '7' : [], '8' : [], '9' : [], '10' : [], '11' : [], '12' : [], '13' : [], '14' : [], '15' : [], '16' : [], '17' : [], '18' : [], '19' : [], '20' : [], 'label' : []})
    df3 = pd.DataFrame({'0' : [], '1' : [], '2' : [], '3' : [], '4' : [], '5' : [], '6' : [], '7' : [], '8' : [], '9' : [], '10' : [], '11' : [], '12' : [], '13' : [], '14' : [], '15' : [], '16' : [], '17' : [], '18' : [], '19' : [], '20' : [], 'label' : []})

    for x, y in tqdm(generator_1.take(n_iterations_1)):
        row = list(x.values()) + [y]
        df1.loc[len(df1.index)] = row
    for x, y in tqdm(generator_2.take(n_iterations_2)):
        row = list(x.values()) + [y]
        df2.loc[len(df2.index)] = row
    for x, y in tqdm(generator_3.take(n_iterations_3)):
        row = list(x.values()) + [y]
        df3.loc[len(df3.index)] = row

    ## Now, I try to imbalance those datasets with the weights given in the paper. 
    ## To do that, I just delete a part of the rows. 
    ## The main problem is that the initial datasets are not balanced at all and I often have to delete a lot of rows to obtain the good weights.
    ## The more I delete, the more rows I have to generate in the first place. (and it depends on the type of generator)
    count1 = df1['label'].value_counts()
    count2 = df2['label'].value_counts()
    count3 = df3['label'].value_counts()

    class_weights_1 = {0: 1, 1: 4/5, 2: 3/5, 3: 2/5, 4: 1/5}
    class_weights_2 = {0: 1, 1: 5/10, 2: 5/10, 3: 4/10, 4: 1/10}
    class_weights_3 = {0: 1, 1: 5/10, 2: 5/10, 3: 4/10, 4: 1/10}

    ## Here is a trick to delete as less rows as possible
    base_1 = min(count1[0], count1[1]*5/4, count1[2]*5/3, count1[3]*5/2, count1[4]*5)
    base_2 = min(count2[0], count2[1]*10/5, count2[2]*10/5, count2[3]*10/4, count2[4]*10/1)
    base_3 = min(count3[0], count3[1]*10/5, count3[2]*10/5, count3[3]*10/4, count3[4]*10/1)

    for class_label, weight in class_weights_1.items():
        # Calculate the number of rows to keep for each class based on the weights
        num_rows_to_keep = int(base_1 * weight)
        # Keep the desired number of rows for each class
        df1[df1['label'] == class_label] = df1[df1['label'] == class_label].sample(n=num_rows_to_keep, random_state=42)
    for class_label, weight in class_weights_2.items():
        # Calculate the number of rows to keep for each class based on the weight
        num_rows_to_keep = int(base_2 * weight)
        # Keep the desired number of rows for each class
        df2[df2['label'] == class_label] = df2[df2['label'] == class_label].sample(n=num_rows_to_keep, random_state=42)
    for class_label, weight in class_weights_3.items():
        # Calculate the number of rows to keep for each class based on the weight
        num_rows_to_keep = int(base_3 * weight)
        # Keep the desired number of rows for each class
        df3[df3['label'] == class_label] = df3[df3['label'] == class_label].sample(n=num_rows_to_keep, random_state=42)
        
    df1 = df1.iloc[:100000] ## I keep the desired number of iterations (given in the paper)
    df2 = df2.iloc[:100000] ## I keep the desired number of iterations (given in the paper)
    df3 = df3.iloc[:100000] ## I keep the desired number of iterations (given in the paper)
    df_weighted = pd.concat([df1, df2, df3])
    df_weighted = df_weighted.dropna(axis=0)
    
    return df_weighted


rbf = RandomRBF(seed_model=1, seed_sample=1, n_classes=5, n_features=21)
rbfdrift = RandomRBFDrift(seed_model=1, seed_sample=1, n_classes=5, n_features=21)
tree = RandomTree(seed_tree=1, seed_sample=1, n_classes=5, n_num_features=21, n_cat_features=0)

df = create_dataset(n_iterations_1= 900000 , n_iterations_2 = 1000000, n_iterations_3 = 350000, generator_1 = rbf, generator_2 = rbfdrift, generator_3 = tree)

df.to_csv('first_synth_data.csv')
