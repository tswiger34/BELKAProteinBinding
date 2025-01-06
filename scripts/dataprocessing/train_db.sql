CREATE TABLE [TrainSMILES] (
  [MoleculeID] INTEGER PRIMARY KEY AUTOINCREMENT,
  [BuildingBlock1_Smiles] TEXT,
  [BuildingBlock2_Smiles] TEXT,
  [BuildingBlock3_Smiles] TEXT,
  [FullMolecule_Smiles] TEXT UNIQUE,
  [BindsEPH] INT,
  [BindsBRD] INT,
  [BindsALB] INT
);

/*

CREATE TABLE Train2dDescriptors(
    ID INT PRIMARY KEY
);

CREATE TABLE Train3dDescriptors(
    ID INT PRIMARY KEY
);

CREATE TABLE BB1_MACCS(
    ID INT PRIMARY KEY
);

CREATE TABLE BB2_MACCS(
    ID INT PRIMARY KEY
);

CREATE TABLE BB3_MACCS(
    ID INT PRIMARY KEY
);

CREATE TABLE TrainFingerprints(
    ID INT PRIMARY KEY
);
*/