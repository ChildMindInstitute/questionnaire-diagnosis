bql_program = [
    '''
        CREATE POPULATION FOR questionnaire_data WITH SCHEMA (
            GUESS STATTYPES OF (*);
            -- stuff that the guess suggested to ignore:
            SET STATTYPES OF
                 "SCQ_30",
                 "SCQ_01",
                 "SCQ_28",
                 "Anxiety",
                 "OtherDx",
                 "UseDisorders",
                 "Compulsions",
                 "SymptomsOfCruelty",
                 "SymptomsOfSuicide"
            TO
                NOMINAL;
            SET STATTYPES OF
                 "Age"
            TO
                NUMERICAL;
            IGNORE
                 "EID",
                 -- ignore components of cruelty and suicide.
                 "CBCL_15",
                 "CBCL_16",
                 "CBCL_26",
                 "CDI2_08",
                 "CBCL_18",
                 "CBCL_91",
                 "MFQ_P_16",
                 "MFQ_P_19"
        );
    ''',
    '''
       CREATE GENERATOR FOR "questionnaire_data" USING loom;
    ''',
    '''
       INITIALIZE 60 MODELS IF NOT EXISTS FOR "questionnaire_data";
    ''',
    '''
       ANALYZE "questionnaire_data" FOR 50 ITERATIONS;
    ''',
    '''
       CREATE TABLE dependencies AS
           ESTIMATE DEPENDENCE PROBABILITY AS "depprob"
               FROM PAIRWISE VARIABLES OF questionnaire_data;
    ''',
]
