// Databricks notebook source
import sys.process._ 
// Change Paths of training and test data sets here after importing them t the DataBricks cluster
val trainPath = "/FileStore/tables/gt53scnf1500150901680/train.csv"
val testPath ="/FileStore/tables/2dvnr6yw1500151098538/test.csv"

// COMMAND ----------

import org.apache.spark.ml.stat._
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import spark.implicits._
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrameStatFunctions
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.ml.regression.{LinearRegressionSummary, LinearRegression}
import org.apache.spark.ml.regression.IsotonicRegression
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel

val housingTrain = spark.read.option("header","true"). option("inferSchema","true").csv(trainPath)
val housingTest = spark.read.option("header","true"). option("inferSchema","true").csv(testPath)

val dataTypeList = housingTrain.dtypes.toList

val nullHandlingFrame = housingTrain


val housingTrain32 = nullHandlingFrame.withColumn("MSZoning", regexp_replace(col("MSZoning"), "NA", "RL")) 
                                      .withColumn("LotFrontage", regexp_replace(col("LotFrontage"), "NA", "70")) //replace with avg
                                      .withColumn("LotArea", regexp_replace(col("LotArea"), "NA", "10516")) //replace with avg
                                      .withColumn("Street", regexp_replace(col("MSZoning"), "NA", "Pave")) 
                                      .withColumn("Utilities", regexp_replace(col("Utilities"), "NA", "AllPub")) 
                                      .withColumn("LotConfig", regexp_replace(col("LotConfig"), "NA", "Inside")) 
                                      .withColumn("LandSlope", regexp_replace(col("LandSlope"), "NA", "Gtl")) 
                                      .withColumn("YearRemodAdd", regexp_replace(col("YearRemodAdd"), "NA", "2011")) 
                                      .withColumn("Exterior1st", regexp_replace(col("Exterior1st"), "NA", "VinylSd")) 
                                      .withColumn("Exterior2nd", regexp_replace(col("Exterior2nd"), "NA", "VinylSd")) 
                                      .withColumn("MasVnrArea", regexp_replace(col("MasVnrArea"), "NA", "0")) 
                                      .withColumn("MasVnrType", regexp_replace(col("MasVnrType"), "NA", "None")) 
                                      .withColumn("ExterQual", regexp_replace(col("ExterQual"), "NA", "TA")) 
                                      .withColumn("ExterCond", regexp_replace(col("ExterCond"), "NA", "TA")) 
                                      .withColumn("MSZoning", regexp_replace(col("Foundation"), "NA", "PConc")) 
                                      .withColumn("Heating", regexp_replace(col("Heating"), "NA", "GasA")) 
                                      .withColumn("CentralAir", regexp_replace(col("CentralAir"), "NA", "Yes"))
                                      .withColumn("BsmtFullBath", regexp_replace(col("BsmtFullBath"), "NA", "0"))
                                      .withColumn("BsmtHalfBath", regexp_replace(col("BsmtHalfBath"), "NA", "0"))
                                      .withColumn("FullBath", regexp_replace(col("FullBath"), "NA", "0"))
                                      .withColumn("HalfBath", regexp_replace(col("HalfBath"), "NA", "0"))
                                      .withColumn("KitchenAbvGr", regexp_replace(col("KitchenAbvGr"), "NA", "0"))
                                      .withColumn("KitchenQual", regexp_replace(col("KitchenQual"), "NA", "TA"))
                                      .withColumn("Functional", regexp_replace(col("Functional"), "NA", "Typical"))
                                      .withColumn("Fireplaces", regexp_replace(col("Fireplaces"), "NA", "0"))
                                      .withColumn("GarageYrBlt", regexp_replace(col("GarageYrBlt"), "NA", "2011"))
                                      .withColumn("GarageYrBlt", regexp_replace(col("GarageYrBlt"), "2207", "2007")) //Noisy data
                                      .withColumn("GarageCars", regexp_replace(col("GarageCars"), "NA", "2")) //filled with mode value
                                      .withColumn("GarageArea", regexp_replace(col("GarageArea"), "NA", "473")) //fill wid avg
                                      .withColumn("PoolArea", regexp_replace(col("PoolArea"), "NA", "0"))
                                      .withColumn("PoolQC", regexp_replace(col("PoolQC"), "NA", "Nopool"))
                                      .withColumn("Fireplaces", regexp_replace(col("Fireplaces"), "NA", "0"))
                                      .withColumn("Electrical", regexp_replace(col("Electrical"), "NA", "SBrkr"))
                                      .withColumn("SaleType", regexp_replace(col("SaleType"), "NA", "WD"))
                                      .withColumn("YearBuiltNew", when($"YearBuilt" >= 1871 and $"YearBuilt" <= 1899, 1).when($"YearBuilt" >= 1900 and $"YearBuilt" <= 1911, 2).when($"YearBuilt" >= 1912 and $"YearBuilt" <= 1931, 3).when($"YearBuilt" >= 1932 and $"YearBuilt" <= 1951, 4).when($"YearBuilt" >= 1952 and $"YearBuilt" <= 1971, 5).when($"YearBuilt" >= 1972 and $"YearBuilt" <= 1991, 6).when($"YearBuilt" >= 1992 and $"YearBuilt" <= 2011, 7))
                                      


val dataTypeList32 = housingTrain32.dtypes.toList

val housingCasted = housingTrain32.withColumn("LotFrontage", 'LotFrontage.cast(IntegerType))
              .withColumn("LotArea", 'LotArea.cast(IntegerType))
              .withColumn("YearRemodAdd", 'YearRemodAdd.cast(IntegerType))
              .withColumn("MasVnrArea", 'MasVnrArea.cast(IntegerType))
              .withColumn("LotArea", 'LotArea.cast(IntegerType))
              .withColumn("BsmtFullBath", 'BsmtFullBath.cast(IntegerType))
              .withColumn("BsmtHalfBath", 'BsmtHalfBath.cast(IntegerType))
              .withColumn("FullBath", 'FullBath.cast(IntegerType))
              .withColumn("HalfBath", 'HalfBath.cast(IntegerType))
              .withColumn("KitchenAbvGr", 'KitchenAbvGr.cast(IntegerType))
              .withColumn("GarageYrBlt", 'GarageYrBlt.cast(IntegerType))
              .withColumn("GarageCars", 'GarageCars.cast(IntegerType))
              .withColumn("GarageArea", 'GarageArea.cast(IntegerType))
              .withColumn("PoolArea", 'PoolArea.cast(IntegerType))
              .withColumn("Fireplaces", 'Fireplaces.cast(IntegerType))


val housingCastedM = housingCasted.col("SalePrice")
val logOfSalePrice = log1p(housingCastedM)
val logOfLotArea = log1p(housingCasted.col("LotArea"))
val logOfGrLivArea = log1p(housingCasted.col("GrLivArea"))
val logOfBsmtFinSF1 = log1p(housingCasted.col("BsmtFinSF1"))
val logOfBsmtFinSF2 = log1p(housingCasted.col("BsmtFinSF2"))
val logOfBsmtUnfSF = log1p(housingCasted.col("BsmtUnfSF"))
val logOfTotalBsmtSF = log1p(housingCasted.col("TotalBsmtSF"))
val logOfGarageArea = log1p(housingCasted.col("GarageArea"))
val logOfWoodDeckSF = log1p(housingCasted.col("WoodDeckSF"))
val logOfOpenPorchSF = log1p(housingCasted.col("OpenPorchSF"))
val logOf3SsnPorch = log1p(housingCasted.col("3SsnPorch"))
val logOfScreenPorch = log1p(housingCasted.col("ScreenPorch"))
val logOfEnclosedPorch = log1p(housingCasted.col("EnclosedPorch"))
val logOfMasVnrArea = log1p(housingCasted.col("MasVnrArea"))
val logOfLotFrontage = log1p(housingCasted.col("LotFrontage"))
val logOf1stFlrSF = log1p(housingCasted.col("1stFlrSF"))
val logOf2ndFlrSF = log1p(housingCasted.col("2ndFlrSF"))
val logOfLowQualFinSF = log1p(housingCasted.col("LowQualFinSF"))
val logOfPoolArea = log1p(housingCasted.col("PoolArea"))
val logOfMiscVal = log1p(housingCasted.col("MiscVal"))
val YearSoldN = housingCasted.col("YrSold").minus(2010).multiply(-1)

val dataTypeListCasted = housingCasted.dtypes.toList

val housingCastedIdx = housingCasted.withColumn("SalePriceNew", logOfSalePrice)
                                    .withColumn("LotAreaNew",logOfLotArea)
                                    .withColumn("GrLivAreaNew",logOfGrLivArea)
                                    .withColumn("BsmtFinSF1New",logOfBsmtFinSF1)
                                    .withColumn("BsmtFinSF2New",logOfBsmtFinSF2)
                                    .withColumn("BsmtUnfSFNew",logOfBsmtUnfSF)
                                    .withColumn("TotalBsmtSFNew",logOfTotalBsmtSF)
                                    .withColumn("GarageAreaNew",logOfGarageArea)
                                    .withColumn("WoodDeckSFNew",logOfWoodDeckSF)
                                    .withColumn("OpenPorchSFNew",logOfOpenPorchSF)
                                    .withColumn("3SsnPorchNew",logOf3SsnPorch)
                                    .withColumn("ScreenPorchNew",logOfScreenPorch)
                                    .withColumn("EnclosedPorchNew",logOfEnclosedPorch)
                                    .withColumn("MasVnrAreaNew",logOfMasVnrArea)
                                    .withColumn("LotFrontageNew",logOfLotFrontage)
                                    .withColumn("1stFlrSFNew",logOf1stFlrSF)
                                    .withColumn("2ndFlrSFNew",logOf2ndFlrSF)
                                    .withColumn("LowQualFinSFNew",logOfLowQualFinSF)
                                    .withColumn("PoolAreaNew",logOfPoolArea)
                                    .withColumn("MiscValNew",logOfMiscVal)
                                    .withColumn("YearSoldNew",YearSoldN)
                                    .withColumn("GarageYearBuiltNew", when($"GarageYrBlt" >= 1871 and $"GarageYrBlt" <= 1899, 1).when($"GarageYrBlt" >= 1900 and $"GarageYrBlt" <= 1911, 2).when($"GarageYrBlt" >= 1912 and $"GarageYrBlt" <= 1931, 3).when($"GarageYrBlt" >= 1932 and $"YearBuilt" <= 1951, 4).when($"GarageYrBlt" >= 1952 and $"GarageYrBlt" <= 1971, 5).when($"GarageYrBlt" >= 1972 and $"GarageYrBlt" <= 1991, 6).when($"GarageYrBlt" >= 1992 and $"GarageYrBlt" <= 2011, 7))
                                    .withColumn("YearRemodelledNew", when($"YearRemodAdd" >= 1871 and $"YearRemodAdd" <= 1899, 1).when($"YearRemodAdd" >= 1900 and $"YearRemodAdd" <= 1911, 2).when($"YearRemodAdd" >= 1912 and $"YearRemodAdd" <= 1931, 3).when($"YearRemodAdd" >= 1932 and $"YearRemodAdd" <= 1951, 4).when($"YearRemodAdd" >= 1952 and $"YearRemodAdd" <= 1971, 5).when($"YearRemodAdd" >= 1972 and $"YearRemodAdd" <= 1991, 6).when($"YearRemodAdd" >= 1992 and $"YearRemodAdd" <= 2011, 7))
                                    .withColumn("reconstruction", when($"YrSold" < $"YearRemodAdd" , 1).otherwise(0))
                                    .withColumn("BuildEquivalent", when($"YrSold" < $"YearBuilt" , 1).otherwise(0))
                                    .drop($"MiscVal")
                                    .drop($"YearRemodAdd")
                                    .drop($"GarageYrBlt")
                                    .drop($"YrSold")
                                    .drop($"SalePrice")
                                    .drop($"PoolQC")
                                    .drop($"Alley")
                                    .drop($"Fence")
                                    .drop($"MiscFeautre")
                                    .drop($"Street")
                                    .drop($"Heating")
                                    .drop($"Condition2")
                                    .drop($"LotArea")
                                    .drop($"GrLivArea")
                                    .drop($"BsmtFinSF1")
                                    .drop($"BsmtFinSF2")
                                    .drop($"TotalBsmtSF")
                                    .drop($"GarageArea")
                                    .drop($"BsmtUnfSF")
                                    .drop($"EnclosedPorch")
                                    .drop($"WoodDeckSF")
                                    .drop($"OpenPorchSF")
                                    .drop($"3SsnPorch")
                                    .drop($"ScreenPorch")
                                    .drop($"LotFrontage")
                                    .drop($"1stFlrSF")
                                    .drop($"2ndFlrSF")
                                    .drop($"LowQualFinSF")
                                    .drop($"PoolArea")
                                    .drop($"MasVnrArea")
                                    .drop($"YearBuilt")

val categoricalAttributes = housingCastedIdx.dtypes.filter(_._2 == "StringType") map (_._1)
val nominalAttributes = housingCastedIdx.dtypes.filter(_._2 == "IntegerType") map (_._1)
val indexers = categoricalAttributes.map( c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_idx"))

println(categoricalAttributes.length)
println(nominalAttributes.length)

val pipelineIndex = new Pipeline().setStages(indexers)
val transformedIndex = pipelineIndex.fit(housingCastedIdx).transform(housingCastedIdx)
val columnDroppedIdx = categoricalAttributes.toList
val filteredDFIdx = transformedIndex.select(transformedIndex.columns .filter(colName => !columnDroppedIdx.contains(colName)) .map(colName => new Column(colName)): _*)
val colsIdx= filteredDFIdx.dtypes
val colnamesIdx = colsIdx.filter(x => x._1 != "SalePriceNew").filter(x =>x._1!= "Id").map(_._1)


val tempIdx = Seq("SalePriceNew")
val labelColumnIdx = filteredDFIdx.select(filteredDFIdx("SalePriceNew").as("label"), $"Id")
val restOfColumnsIdx = filteredDFIdx.select(filteredDFIdx.columns .filter(colName => !tempIdx.contains(colName)) .map(colName => new Column(colName)): _*) 
val joinedDFIdx = labelColumnIdx.join(restOfColumnsIdx, "Id")
val finalDFIdx = joinedDFIdx.drop($"Id")
                            
                          

val features =new VectorAssembler().setInputCols(colnamesIdx).setOutputCol("features")
val output = features.transform(finalDFIdx)
val regModelData = output.select("label", "features")

val target = regModelData.columns.indexOf("label")
//println(regModelData)

// COMMAND ----------

val nullHandlingTest = housingTest

val housingTesthandled = nullHandlingTest.withColumn("MSZoning", regexp_replace(col("MSZoning"), "NA", "RL")) 
                                      .withColumn("LotFrontage", regexp_replace(col("LotFrontage"), "NA", "70")) //replace with avg
                                      .withColumn("LotArea", regexp_replace(col("LotArea"), "NA", "10516")) //replace with avg
                                      .withColumn("Street", regexp_replace(col("MSZoning"), "NA", "Pave")) 
                                      .withColumn("Utilities", regexp_replace(col("Utilities"), "NA", "AllPub")) 
                                      .withColumn("LotConfig", regexp_replace(col("LotConfig"), "NA", "Inside")) 
                                      .withColumn("LandSlope", regexp_replace(col("LandSlope"), "NA", "Gtl")) 
                                      .withColumn("YearRemodAdd", regexp_replace(col("YearRemodAdd"), "NA", "2011")) 
                                      .withColumn("Exterior1st", regexp_replace(col("Exterior1st"), "NA", "VinylSd")) 
                                      .withColumn("Exterior2nd", regexp_replace(col("Exterior2nd"), "NA", "VinylSd")) 
                                      .withColumn("MasVnrArea", regexp_replace(col("MasVnrArea"), "NA", "0")) 
                                      .withColumn("MasVnrType", regexp_replace(col("MasVnrType"), "NA", "None")) 
                                      .withColumn("ExterQual", regexp_replace(col("ExterQual"), "NA", "TA")) 
                                      .withColumn("ExterCond", regexp_replace(col("ExterCond"), "NA", "TA")) 
                                      .withColumn("MSZoning", regexp_replace(col("Foundation"), "NA", "PConc")) 
                                      .withColumn("Heating", regexp_replace(col("Heating"), "NA", "GasA")) 
                                      .withColumn("CentralAir", regexp_replace(col("CentralAir"), "NA", "Yes"))
                                      .withColumn("BsmtFullBath", regexp_replace(col("BsmtFullBath"), "NA", "0"))
                                      .withColumn("BsmtHalfBath", regexp_replace(col("BsmtHalfBath"), "NA", "0"))
                                      .withColumn("FullBath", regexp_replace(col("FullBath"), "NA", "0"))
                                      .withColumn("HalfBath", regexp_replace(col("HalfBath"), "NA", "0"))
                                      .withColumn("KitchenAbvGr", regexp_replace(col("KitchenAbvGr"), "NA", "0"))
                                      .withColumn("KitchenQual", regexp_replace(col("KitchenQual"), "NA", "TA"))
                                      .withColumn("Functional", regexp_replace(col("Functional"), "NA", "Typical"))
                                      .withColumn("Fireplaces", regexp_replace(col("Fireplaces"), "NA", "0"))
                                      .withColumn("GarageYrBlt", regexp_replace(col("GarageYrBlt"), "NA", "2011"))
                                      .withColumn("GarageYrBlt", regexp_replace(col("GarageYrBlt"), "2207", "2007"))
                                      .withColumn("GarageCars", regexp_replace(col("GarageCars"), "NA", "2"))
                                      .withColumn("GarageArea", regexp_replace(col("GarageArea"), "NA", "473")) //fill wid avg
                                      .withColumn("PoolArea", regexp_replace(col("PoolArea"), "NA", "0"))
                                      .withColumn("PoolQC", regexp_replace(col("PoolQC"), "NA", "Nopool"))
                                      .withColumn("Fireplaces", regexp_replace(col("Fireplaces"), "NA", "0"))
                                      .withColumn("Electrical", regexp_replace(col("Electrical"), "NA", "SBrkr"))
                                      .withColumn("BsmtFinSF1", regexp_replace(col("BsmtFinSF1"), "NA", "0"))
                                      .withColumn("BsmtFinSF2", regexp_replace(col("BsmtFinSF2"), "NA", "0"))
                                      .withColumn("BsmtUnfSF", regexp_replace(col("BsmtUnfSF"), "NA", "0"))
                                      .withColumn("TotalBsmtSF", regexp_replace(col("TotalBsmtSF"), "NA", "0"))
                                      .withColumn("SaleType", regexp_replace(col("SaleType"), "NA", "WD"))
                                      .withColumn("YearBuiltNew", when($"YearBuilt" >= 1871 and $"YearBuilt" <= 1899, 1).when($"YearBuilt" >= 1900 and $"YearBuilt" <= 1911, 2).when($"YearBuilt" >= 1912 and $"YearBuilt" <= 1931, 3).when($"YearBuilt" >= 1932 and $"YearBuilt" <= 1951, 4).when($"YearBuilt" >= 1952 and $"YearBuilt" <= 1971, 5).when($"YearBuilt" >= 1972 and $"YearBuilt" <= 1991, 6).when($"YearBuilt" >= 1992 and $"YearBuilt" <= 2011, 7))

                                      
val housingCastedTest = housingTesthandled.withColumn("LotFrontage", 'LotFrontage.cast(IntegerType))
              .withColumn("BsmtFinSF1",'BsmtFinSF1.cast(IntegerType))
              .withColumn("BsmtFinSF2",'BsmtFinSF2.cast(IntegerType))
              .withColumn("BsmtUnfSF",'BsmtUnfSF.cast(IntegerType))
              .withColumn("TotalBsmtSF",'TotalBsmtSF.cast(IntegerType))
              .withColumn("LotArea", 'LotArea.cast(IntegerType))
              .withColumn("YearRemodAdd", 'YearRemodAdd.cast(IntegerType))
              .withColumn("MasVnrArea", 'MasVnrArea.cast(IntegerType))
              .withColumn("LotArea", 'LotArea.cast(IntegerType))
              .withColumn("BsmtFullBath", 'BsmtFullBath.cast(IntegerType))
              .withColumn("BsmtHalfBath", 'BsmtHalfBath.cast(IntegerType))
              .withColumn("FullBath", 'FullBath.cast(IntegerType))
              .withColumn("HalfBath", 'HalfBath.cast(IntegerType))
              .withColumn("KitchenAbvGr", 'KitchenAbvGr.cast(IntegerType))
              .withColumn("GarageYrBlt", 'GarageYrBlt.cast(IntegerType))
              .withColumn("GarageCars", 'GarageCars.cast(IntegerType))
              .withColumn("GarageArea", 'GarageArea.cast(IntegerType))
              .withColumn("PoolArea", 'PoolArea.cast(IntegerType))
              .withColumn("Fireplaces", 'Fireplaces.cast(IntegerType))

val logOfLotAreaT = log1p(housingCastedTest.col("LotArea"))
val logOfGrLivAreaT = log1p(housingCastedTest.col("GrLivArea"))
val logOfBsmtFinSF1T = log1p(housingCastedTest.col("BsmtFinSF1"))
val logOfBsmtFinSF2T = log1p(housingCastedTest.col("BsmtFinSF2"))
val logOfBsmtUnfSFT = log1p(housingCastedTest.col("BsmtUnfSF"))
val logOfTotalBsmtSFT = log1p(housingCastedTest.col("TotalBsmtSF"))
val logOfGarageAreaT = log1p(housingCastedTest.col("GarageArea"))
val logOfWoodDeckSFT = log1p(housingCastedTest.col("WoodDeckSF"))
val logOfOpenPorchSFT = log1p(housingCastedTest.col("OpenPorchSF"))
val logOf3SsnPorchT = log1p(housingCastedTest.col("3SsnPorch"))
val logOfScreenPorchT = log1p(housingCastedTest.col("ScreenPorch"))
val logOfEnclosedPorchT = log1p(housingCastedTest.col("EnclosedPorch"))
val logOfMasVnrAreaT = log1p(housingCastedTest.col("MasVnrArea"))
val logOfLotFrontageT = log1p(housingCastedTest.col("LotFrontage"))
val logOf1stFlrSFT = log1p(housingCastedTest.col("1stFlrSF"))
val logOf2ndFlrSFT = log1p(housingCastedTest.col("2ndFlrSF"))
val logOfLowQualFinSFT = log1p(housingCastedTest.col("LowQualFinSF"))
val logOfPoolAreaT = log1p(housingCastedTest.col("PoolArea"))
val logOfMiscValT = log1p(housingCastedTest.col("MiscVal"))
val YearSoldNT = housingCastedTest.col("YrSold").minus(2010).multiply(-1)
              
val housingT = housingCastedTest.withColumn("LotAreaNew",logOfLotAreaT)
                                    .withColumn("GrLivAreaNew",logOfGrLivAreaT)
                                    .withColumn("BsmtFinSF1New",logOfBsmtFinSF1T)
                                    .withColumn("BsmtFinSF2New",logOfBsmtFinSF2T)
                                    .withColumn("BsmtUnfSFNew",logOfBsmtUnfSFT)
                                    .withColumn("TotalBsmtSFNew",logOfTotalBsmtSFT)
                                    .withColumn("GarageAreaNew",logOfGarageAreaT)
                                    .withColumn("WoodDeckSFNew",logOfWoodDeckSFT)
                                    .withColumn("OpenPorchSFNew",logOfOpenPorchSFT)
                                    .withColumn("3SsnPorchNew",logOf3SsnPorchT)
                                    .withColumn("ScreenPorchNew",logOfScreenPorchT)
                                    .withColumn("EnclosedPorchNew",logOfEnclosedPorchT)
                                    .withColumn("MasVnrAreaNew",logOfMasVnrAreaT)
                                    .withColumn("LotFrontageNew",logOfLotFrontageT)
                                    .withColumn("1stFlrSFNew",logOf1stFlrSFT)
                                    .withColumn("2ndFlrSFNew",logOf2ndFlrSFT)
                                    .withColumn("LowQualFinSFNew",logOfLowQualFinSFT)
                                    .withColumn("PoolAreaNew",logOfPoolAreaT)
                                    .withColumn("MiscValNew",logOfMiscValT)
                                    .withColumn("YearSoldNew",YearSoldNT)
                                    .withColumn("GarageYearBuiltNew", when($"GarageYrBlt" >= 1871 and $"GarageYrBlt" <= 1899, 1).when($"GarageYrBlt" >= 1900 and $"GarageYrBlt" <= 1911, 2).when($"GarageYrBlt" >= 1912 and $"GarageYrBlt" <= 1931, 3).when($"GarageYrBlt" >= 1932 and $"YearBuilt" <= 1951, 4).when($"GarageYrBlt" >= 1952 and $"GarageYrBlt" <= 1971, 5).when($"GarageYrBlt" >= 1972 and $"GarageYrBlt" <= 1991, 6).when($"GarageYrBlt" >= 1992 and $"GarageYrBlt" <= 2011, 7))
                                    .withColumn("YearRemodelledNew", when($"YearRemodAdd" >= 1871 and $"YearRemodAdd" <= 1899, 1).when($"YearRemodAdd" >= 1900 and $"YearRemodAdd" <= 1911, 2).when($"YearRemodAdd" >= 1912 and $"YearRemodAdd" <= 1931, 3).when($"YearRemodAdd" >= 1932 and $"YearRemodAdd" <= 1951, 4).when($"YearRemodAdd" >= 1952 and $"YearRemodAdd" <= 1971, 5).when($"YearRemodAdd" >= 1972 and $"YearRemodAdd" <= 1991, 6).when($"YearRemodAdd" >= 1992 and $"YearRemodAdd" <= 2011, 7))
                                    .withColumn("reconstruction", when($"YrSold" < $"YearRemodAdd" , 1).otherwise(0))
                                    .withColumn("BuildEquivalent", when($"YrSold" < $"YearBuilt" , 1).otherwise(0))
                                    .drop($"MiscVal")
                                    .drop($"YearBuilt")
                                    .drop($"YrSold")
                                    .drop($"YearRemodAdd")
                                    .drop($"GarageYrBlt")
                                    .drop($"PoolQC")
                                    .drop($"Alley")
                                    .drop($"Fence")
                                    .drop($"MiscFeautre")
                                    .drop($"Street")
                                    .drop($"Heating")
                                    .drop($"Condition2")
                                    .drop($"LotArea")
                                    .drop($"GrLivArea")
                                    .drop($"BsmtFinSF1")
                                    .drop($"BsmtFinSF2")
                                    .drop($"TotalBsmtSF")
                                    .drop($"GarageArea")
                                    .drop($"BsmtUnfSF")
                                    .drop($"EnclosedPorch")
                                    .drop($"WoodDeckSF")
                                    .drop($"OpenPorchSF")
                                    .drop($"3SsnPorch")
                                    .drop($"ScreenPorch")
                                    .drop($"LotFrontage")
                                    .drop($"1stFlrSF")
                                    .drop($"2ndFlrSF")
                                    .drop($"LowQualFinSF")
                                    .drop($"PoolArea")
                                    .drop($"MasVnrArea")


val categoricalAttributesT = housingT.dtypes.filter(_._2 == "StringType") map (_._1)
val nominalAttributesT = housingT.dtypes.filter(_._2 == "IntegerType") map (_._1)
val indexersT = categoricalAttributesT.map( c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_idx"))


val pipelineIndexT = new Pipeline().setStages(indexersT)
val transformedIndexT = pipelineIndexT.fit(housingT).transform(housingT)
val columnDroppedIdxT = categoricalAttributesT.toList
val filteredDFIdxT = transformedIndexT.select(transformedIndexT.columns .filter(colName => !columnDroppedIdxT.contains(colName)) .map(colName => new Column(colName)): _*)
val colsIdxT = filteredDFIdxT.dtypes
val colnamesIdxT = colsIdxT.filter(x =>x._1!= "Id").map(_._1)

val finalDFIdxT = filteredDFIdxT.drop($"Id")
  
val featuresT =new VectorAssembler().setInputCols(colnamesIdxT).setOutputCol("features")
val outputT = featuresT.transform(finalDFIdxT)
val regModelDataT = outputT.select("features")


// COMMAND ----------

def addIndex(df: DataFrame) = sqlContext.createDataFrame(
  
  df.rdd.zipWithIndex.map{case (r, i) => Row.fromSeq(r.toSeq :+ i)},
  
  StructType(df.schema.fields :+ StructField("_index", LongType, false))
)

val idIdx = filteredDFIdxT.select($"Id")

val regModelWithIndex = addIndex(regModelDataT)
val idIdxWithIndex = addIndex(idIdx)

val joinedRegModel = regModelWithIndex.join(idIdxWithIndex, Seq("_index")).sort("_index").drop("_index")

// COMMAND ----------

//************************************** Implementation of Linear Regression and Lasso Regression *******************************************************************//

// COMMAND ----------

// ********************************** Linear Regression and Lasso Regression TrainingValidationSplit (Train rmse, r2) *************************************************//

import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator

val Array(training, test) = regModelData.randomSplit(Array(0.9, 0.1), seed = 1)

val lr = new LinearRegression()
    .setMaxIter(100)

val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.1, 0.01, 0.05, 1))
  .addGrid(lr.fitIntercept)
  .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0, 0.0001, 0.0005, 0.00075))
  .build()

val trainValidationSplit = new TrainValidationSplit()
  .setEstimator(lr)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setTrainRatio(0.8)

val runmodel = trainValidationSplit.fit(training)



runmodel.transform(test)
  .select("features", "label", "prediction")
  .show()
  

val gridSummary = runmodel.validationMetrics.zip(runmodel.getEstimatorParamMaps).sortBy(-_._1)
gridSummary.foreach { case (metric, params) =>
        println(metric)
        println(params)
        println()
    }

// COMMAND ----------

/* ********************************** Linear Regression and Lasso Regression Cross Validation and Generating Submission File (Train rmse, r2) ********************** ******************************************************************************************************************************************************************** */

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

val lr = new LinearRegression()
    .setMaxIter(100)
    
val pipeline = new Pipeline().setStages(Array(lr))

val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.01))
  .addGrid(lr.fitIntercept, Array(true))
  .addGrid(lr.elasticNetParam, Array(0.5))
  .build()

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)

val cvModel = cv.fit(regModelData)

val cvModelPred = cvModel.transform(regModelDataT)
val actualPredtemp = cvModelPred.col("prediction")
val actualPred = expm1(actualPredtemp)
val resultPred = cvModelPred.withColumn("predictionActual",actualPred).drop($"prediction")
val finalPred = resultPred.join(joinedRegModel, "features").drop($"features")
val submission_csv = finalPred.select(finalPred("Id"),finalPred("predictionActual").as("SalePrice"))

// COMMAND ----------

display(submission_csv) 

// COMMAND ----------

// ************************************************* Implementation of Random Forest Regression *************************************************************//

// COMMAND ----------

// ********************************** Random Forest Regression TrainingValidationSplit (Train rmse, r2) *************************************************//

import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator

//val Array(trainingRF, testRF) = regModelData.randomSplit(Array(0.8, 0.2), seed = 12345)

val rf = new RandomForestRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setNumTrees(500)
//  .setMaxDepth(10)
//  .setMaxBins(48)

val pipelineRF = new Pipeline().setStages(Array(rf))
val Array(trainingDataRF, testingDataRF) = regModelData.randomSplit(Array(0.75, 0.25), seed = 1)
trainingDataRF.schema.fields.foreach(println)


val paramGridRF = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(2, 5, 10, 30, 50 ,100, 250, 500))
  .addGrid(rf.maxDepth,Array(4))
  //.addGrid(rf.maxBins, Array(120,140,160))
  .build()

val trainValidationSplitRF = new TrainValidationSplit()
  .setEstimator(rf)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGridRF)
  .setTrainRatio(0.8)

val modelRF = trainValidationSplitRF.fit(trainingDataRF)


//val modelRF = pipelineRF.fit(trainingDataRF)
val predictionsRF = modelRF.transform(testingDataRF)
predictionsRF.select("prediction", "label", "features").show()

val gridSummaryRF = modelRF.validationMetrics.zip(modelRF.getEstimatorParamMaps).sortBy(-_._1)
gridSummaryRF.foreach { case (metric, params) =>
        println(metric)
        println(params)
        println()
    }

val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictionsRF)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val evaluatorRF2 = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("r2")
val r2rf2 = evaluatorRF2.evaluate(predictionsRF)

println("R2  on test data = " + r2rf2)

// COMMAND ----------

// ************************************************* Random Forest Regression Generating Submission File *****************************************************************//

import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator

val rf = new RandomForestRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setNumTrees(250)

val pipelineRF = new Pipeline().setStages(Array(rf))

val modelRF = pipelineRF.fit(regModelData)

val predictionsRF = modelRF.transform(regModelDataT)

predictionsRF.select("prediction", "features").show()

val actualPredtempRF = predictionsRF.col("prediction")
val actualPredRF = expm1(actualPredtempRF)
val resultPredRF = predictionsRF.withColumn("predictionActual",actualPredRF).drop($"prediction")
val finalPredRF = resultPredRF.join(joinedRegModel, "features").drop($"features")
val submission_csvRF = finalPredRF.select(finalPredRF("Id"),finalPredRF("predictionActual").as("SalePrice"))


// COMMAND ----------

display(submission_csvRF)

// COMMAND ----------

// ********************************************** Implementation of Gradient Boosting Tree Regression **************************************************************//

// COMMAND ----------

// ********************************** Gradient Boosting Tree Regression TrainingValidationSplit (Train rmse, r2) *************************************************//
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

val Array(trainingDataGBT, testDataGBT) = regModelData.randomSplit(Array(0.7, 0.3))
val gbt = new GBTRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")
  //.setNumIterations(100)
  //.setMaxIter(10)

//val pipelineGBT = new Pipeline().setStages(Array(gbt))


val paramGridGBT = new ParamGridBuilder()
  .addGrid(gbt.minInfoGain, Array(0.1, 0.5, 0.01, 0.05, 0.0))
  .addGrid(gbt.maxDepth,Array(5, 8))
  .addGrid(gbt.stepSize,Array(0.1, 0.05, 0.5))
  .build()

val trainValidationSplitGBT = new TrainValidationSplit()
  .setEstimator(gbt)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGridGBT)
  .setTrainRatio(0.8)
val modelGBT = trainValidationSplitGBT.fit(trainingDataGBT)
val predictionsGBT = modelGBT.transform(testDataGBT)

predictionsGBT.select("prediction", "label", "features").show(5)

val gridSummaryGBT = modelGBT.validationMetrics.zip(modelGBT.getEstimatorParamMaps).sortBy(-_._1)
gridSummaryGBT.foreach { case (metric, params) =>
        println(metric)
        println(params)
        println()
    }

val evaluatorGBT = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val evaluatorGBT2 = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("r2")
val r2GBT2 = evaluatorGBT2.evaluate(predictionsGBT)
println("R2  on test data = " + r2GBT2)

val rmseGBT = evaluatorGBT.evaluate(predictionsGBT)
println("Root Mean Squared Error (RMSE) on test data = " + rmseGBT)

// COMMAND ----------

// ********************************** Gradient Boosting Tree Regression Generating submission file for best parameters *************************************************//

import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

val gbt = new GBTRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMinInfoGain(0.0)
  .setMaxDepth(5)
  .setStepSize(0.1)
  .setMaxIter(100)

val pipelineGBT = new Pipeline().setStages(Array(gbt))

val modelGBT = pipelineGBT.fit(regModelData)

val predictionsGBT = modelGBT.transform(regModelDataT)

predictionsGBT.select("prediction", "features").show()

val actualPredtempGBT = predictionsGBT.col("prediction")
val actualPredGBT = expm1(actualPredtempGBT)
val resultPredGBT = predictionsGBT.withColumn("predictionActual",actualPredGBT).drop($"prediction")
val finalPredGBT = resultPredGBT.join(joinedRegModel, "features").drop($"features")
val submission_csvGBT = finalPredGBT.select(finalPredGBT("Id"),finalPredGBT("predictionActual").as("SalePrice"))


// COMMAND ----------

display(submission_csvGBT)
