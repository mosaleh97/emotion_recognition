#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    # Download input artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Log a message indicating that the artifact was downloaded
    logger.info(f"Downloaded {args.input_artifact} from W&B")

    # Load the artifact as a pandas dataframe
    df = pd.read_csv(artifact_local_path)
    
    # Remove outliers from price column out of the range [min_price, max_price]
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    #Drop rows with missing values
    df.dropna(inplace=True)

    # Dataset is now cleaned
    logger.info(f"Dataset cleaned")

    # Save the cleaned dataframe to a csv file
    df.to_csv('cleaned_data.csv', index=False)

    # Log the cleaned data as an artifact-wandb.Artifact params: name, type, description
    artifact = wandb.Artifact("cleaned_data", type=args.output_type, description=args.output_description)

    # Add a file to the artifact
    artifact.add_file('cleaned_data.csv')

    # Log the artifact to W&B
    run.log_artifact(artifact)

    # Log a message indicating that the artifact was uploaded
    logger.info(f"Uploaded {args.output_artifact} to W&B")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type= str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to consider",
        required=True
    )


    args = parser.parse_args()

    go(args)
