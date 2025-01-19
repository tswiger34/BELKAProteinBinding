CREATE TABLE [TestSMILES] (
  [ObsID] INTEGER PRIMARY KEY AUTOINCREMENT,
  [BuildingBlock1_Smiles] TEXT,
  [BuildingBlock2_Smiles] TEXT,
  [BuildingBlock3_Smiles] TEXT,
  [FullMolecule_Smiles] TEXT,
  [TargetProtein] TEXT
);

/*
CREATE TABLE ProteinTargets (
    ProteinTargetID TINYINT PRIMARY KEY,
    ProteinName VARCHAR(8),
    AF2Confidence REAL,
    AF2Annotations REAL,
    AF2Solvent REAL,
    AF2BindingRegions REAL
);

CREATE TABLE Test2dDescriptors(
    ID INT PRIMARY KEY
);

CREATE TABLE Test3dDescriptors(
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

CREATE TABLE TestFingerprints(
    ID INT PRIMARY KEY
);
*/