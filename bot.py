#!/usr/bin/env python3

import os
import subprocess
import discord
from discord.ext import commands
from discord import app_commands
from discord import Message, Embed
from dotenv import load_dotenv
from groq import Groq, AuthenticationError, RateLimitError
from collections import defaultdict
import requests
from datetime import datetime, timedelta
import google.generativeai as gemini
import textwrap
import traceback
import asyncio
import logging
from bs4 import BeautifulSoup  
from hunger_games import HungerGames, Participant 
import random
import edge_tts
import io
from contextlib import redirect_stdout
import aiohttp
import lyricsgenius
from timeit import default_timer as timer 
from google.oauth2 import service_account
from PIL import Image
import yt_dlp
from enum import Enum
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MVSEP_API_KEY = os.getenv('MVSEP_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') 
GENIUS_API_TOKEN = os.getenv('GENIUS_API_TOKEN')

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def handle_groq_error(e, model_name):
    error_data = e.args[0]

    if isinstance(error_data, str):
        json_match = re.search(r'(\{.*\})', error_data)
        if json_match:
            json_str = json_match.group(1)
            json_str = json_str.replace("'", '"') 
            error_data = json.loads(json_str)

    if isinstance(e, AuthenticationError):
        if isinstance(error_data, dict) and 'error' in error_data and 'message' in error_data['error']:
            error_message = error_data['error']['message']
            raise discord.app_commands.AppCommandError(error_message)
    elif isinstance(e, RateLimitError):
        if isinstance(error_data, dict) and 'error' in error_data and 'message' in error_data['error']:
            error_message = error_data['error']['message']
            error_message = re.sub(r'org_[a-zA-Z0-9]+', 'org_(censored)', error_message) 
            raise discord.app_commands.AppCommandError(error_message)
    else:
        raise discord.app_commands.AppCommandError(f"Error during Groq API call: {e}")

# Initialize the Google Generative AI client
gemini.configure(api_key=GOOGLE_API_KEY)

# Initialize genius shit
genius = lyricsgenius.Genius("GENIUS_API_TOKEN")

# Initialize the bot with intents
intents = discord.Intents.default()
intents.messages = True 
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Bot-wide settings
bot_settings = {
    "model": "llama-3.1-70b-versatile",
    "system_prompt": "You are a helpful and friendly AI assistant.",
    "context_messages": 5,
    "llm_enabled": True  # LLM is enabled by default for the entire bot
}
code_language = "python"
# summary
MESSAGE_COUNT = 0
# Permission system
class PermissionLevel(Enum):
    USER = 1
    MODERATOR = 2
    ADMINISTRATOR = 3

role_permissions = {
    1221039013503303700: PermissionLevel.USER,      
    1266750753255587972: PermissionLevel.MODERATOR,
    1198707036070871102: PermissionLevel.ADMINISTRATOR
}

      
def is_authorized(interaction: discord.Interaction, required_permission_level: PermissionLevel = PermissionLevel.USER):
    """Check if the user has the required permission level to use the command."""
    user = interaction.user
    if user.id in authorized_users:
        return True
    for role in user.roles:
        if role.id in role_permissions and role_permissions[role.id].value >= required_permission_level.value:
            return True
    return False

    
authorized_users = []   # Replace with user IDs
authorized_roles = [1198707036070871102]   # Replace with role IDs

# Define valid model names for Groq and Gemini
groq_models = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma-7b-it",
    "gemma2-9b-it",
    "llava-v1.5-7b-4096-preview",
    "llama-guard-3-8b"
]

# Define valid model names for Gemini
gemini_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-pro-exp-0801"
]

local_models = [  # Renamed from gemini_models
    # Add your local model names here
    "gemma2:2b",
]
    # --- Conversation Data (Important!) chatting shit
conversation_data = defaultdict(lambda: {"messages": []}) 

# --- loggin shit

logging.basicConfig(filename='bot_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



# --- Application Commands ---

@bot.tree.command(name="sync_commands", description="Sync / bot commands (ADMIN ONLY).")
async def sync_commands(interaction: discord.Interaction):
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return
    await bot.tree.sync()
    await interaction.response.send_message("Bot commands synced!", ephemeral=True)

from typing import Optional  

@bot.tree.command(name="edit_profile", description="Edit bot profile (Admin only).")
async def edit_profile(interaction: discord.Interaction,
                  username: Optional[str] = None,
                  app_name: Optional[str] = None,
                  description: Optional[str] = None,  # Add description parameter
                  avatar: Optional[discord.Attachment] = None,
                  banner: Optional[discord.Attachment] = None):
    """
    Edit various bot settings (Admin only).

    Usage: /edit_profile [username] [app_name] [description] [avatar] [banner]

    Example:
    /edit_profile username="New Bot Name"
    """
    await interaction.response.defer(ephemeral=True)

    try:
        updated_fields = []
        if username:
            await bot.user.edit(username=username)
            updated_fields.append(f"Username changed to '{username}'")
        if app_name:
            app_info = await bot.application_info()  # Await the coroutine
            await app_info.edit(name=app_name)
            updated_fields.append(f"Application name changed to '{app_name}'")
        if description:
            app_info = await bot.application_info()  # Await the coroutine
            await app_info.edit(description=description)
            updated_fields.append(f"Description changed to '{description}'")
        if avatar:
            avatar_data = await avatar.read()
            await bot.user.edit(avatar=avatar_data)
            updated_fields.append("Avatar updated")
        if banner:
            banner_data = await banner.read()
            await bot.user.edit(banner=banner_data)
            updated_fields.append("Banner updated")

        if updated_fields:
            await interaction.followup.send("\n".join(updated_fields))
        else:
            await interaction.followup.send("No settings were changed.")

    except discord.HTTPException as e:
        await interaction.followup.send(f"An error occurred: {e}")
        logging.error(f"Error editing bot settings: {e}")

MAX_FILE_SIZE = 100 * 1024 * 1024 
@bot.tree.command(name="image_to_gif", description="convert an image to a gif")
async def image_to_gif(interaction: discord.Interaction, image: discord.Attachment):
    """penis"""
    if not image.content_type.startswith("image/"):
        await interaction.response.send_message("Invalid file type. Please upload an image.", ephemeral=True)
        return
    if image.size > MAX_FILE_SIZE:
        await interaction.response.send_message("Image file size is too large (max 100MB).", ephemeral=True)
        return
    await interaction.response.defer() 
    try:
        image_data = await image.read()
        with Image.open(io.BytesIO(image_data)) as img:
            gif_path = f"temp_{image.filename}.gif"
            img.save(gif_path, save_all=True, optimize=False, duration=100, loop=0)
            with open(gif_path, "rb") as f:
                gif_file = discord.File(f)
                await interaction.followup.send(file=gif_file)
            os.remove(gif_path)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}")
        
@bot.tree.command(name="eval", description="Evaluate Python code. (ADMIN ONLY)")
async def eval_code(interaction: discord.Interaction, code: str):
    """
    Evaluates Python code provided by the bot owner. 

    Features:
     - Code block formatting
     - Standard output capture
     - Error handling and traceback display
     - Result truncation for large outputs
     - Execution time measurement
    """

    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    env = {
        "bot": bot,
        "discord": discord,
        "commands": commands,
        "interaction": interaction,
        "channel": interaction.channel,
        "guild": interaction.guild,
        "message": interaction.message,
    }

    code = code.strip("`")

    stdout = io.StringIO()

    start_time = timer() 
    try:
        with redirect_stdout(stdout):
            exec(
                f"async def func():\n{textwrap.indent(code, '    ')}", env
            )
            result = await env["func"]()  
            result_str = str(result) if result is not None else "No output."

    except Exception as e:

        result_str = "".join(
            traceback.format_exception(type(e), e, e.__traceback__)
        )

    end_time = timer() 
    execution_time = (end_time - start_time) * 1000 


    if len(result_str) > 1900: 
        result_str = result_str[:1900] + "... (Output truncated)"


    await interaction.response.send_message(
        f"```python\n{code}\n```\n**Output:**\n```\n{result_str}\n```\n**Execution time:** {execution_time:.2f} ms",
        ephemeral=True,
    )
    

# whisper related

# Allowed file extensions
ALLOWED_FILE_EXTENSIONS = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
MAX_FILE_SIZE_MB = 25
CHUNK_SIZE_MB = 25

def split_audio(input_file_path, chunk_size_mb):
    chunk_size = chunk_size_mb * 1024 * 1024 
    file_number = 1
    chunks = []
    with open(input_file_path, 'rb') as f:
        chunk = f.read(chunk_size)
        while chunk:
            chunk_name = f"{os.path.splitext(input_file_path)[0]}_part{file_number:03}.mp3"
            with open(chunk_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunks.append(chunk_name)
            file_number += 1
            chunk = f.read(chunk_size)
    return chunks


def check_file(input_file_path):

    file_size_mb = os.path.getsize(input_file_path) / (1024 * 1024)
    file_extension = input_file_path.split(".")[-1].lower()

    if file_extension not in ALLOWED_FILE_EXTENSIONS:
        raise discord.app_commands.AppCommandError(f"Invalid file type (.{file_extension}). Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}")

    if file_size_mb > MAX_FILE_SIZE_MB:
        logging.warning(f"File size too large ({file_size_mb:.2f} MB). Attempting to downsample to 16kHz MP3 128kbps. Maximum size allowed: {MAX_FILE_SIZE_MB} MB")

        output_file_path = os.path.splitext(input_file_path)[0] + "_downsampled.mp3"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    input_file_path,
                    "-ar",
                    "16000",
                    "-ab",
                    "128k",
                    "-ac",
                    "1",
                    "-f",
                    "mp3",
                    "-y",
                    output_file_path,
                ],
                check=True
            )

            downsampled_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
            if downsampled_size_mb > MAX_FILE_SIZE_MB:
                logging.warning(f"File still too large after downsampling ({downsampled_size_mb:.2f} MB). Splitting into {CHUNK_SIZE_MB} MB chunks.")
                return split_audio(output_file_path, CHUNK_SIZE_MB), "split"

            return output_file_path, None
        except subprocess.CalledProcessError as e:
            raise discord.app_commands.AppCommandError(f"Error during downsampling: {e}")
    return input_file_path, None

@bot.tree.command(name="transcript", description="Transcribe an audio file using the Groq Whisper model.")
@app_commands.describe(
    audio_file="The audio file to transcribe.",
    asr_model="The ASR model to use.",
    language="The language of the audio (optional, auto-detect by default).",
    response_format="The format of the transcript.",
    temperature="The sampling temperature, between 0 and 1.",
    prompt="An optional text to guide the model's predictions."
)
@app_commands.choices(
    asr_model=[
        discord.app_commands.Choice(name="Whisper large-v3", value="whisper-large-v3"),
        discord.app_commands.Choice(name="Distil-Whisper English", value="distil-whisper-large-v3-en"),
    ],
    response_format=[
        discord.app_commands.Choice(name="text", value="text"),
        discord.app_commands.Choice(name="json", value="json"),
        discord.app_commands.Choice(name="verbose_json", value="verbose_json"),
    ]
)
async def transcript(interaction: discord.Interaction,
                  audio_file: discord.Attachment,
                  asr_model: str = "whisper-large-v3",
                  language: Optional[str] = None,
                  response_format: Optional[str] = "text",
                  temperature: Optional[float] = 0.0,
                  prompt: Optional[str] = None):

    file_extension = os.path.splitext(audio_file.filename)[1][1:].lower()  # Get extension without dot
    if file_extension not in ALLOWED_FILE_EXTENSIONS:
        await interaction.response.send_message(f"Invalid file type (.{file_extension}). Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}", ephemeral=True)
        return
    
    if not (0.0 <= temperature <= 1.0):
        await interaction.response.send_message(
            "Invalid temperature. Please choose a value between 0.0 and 1.0.", ephemeral=True
        )
        return

    await interaction.response.defer()

    try:
        # Download the audio file
        file_path = f"transcript/{audio_file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        await audio_file.save(file_path)

        processed_path, split_status = check_file(file_path)

        if split_status == "split":
            full_transcript = ""
            for chunk_path in processed_path:
                try:
                    with open(chunk_path, "rb") as file:
                        transcription_response = client.audio.transcriptions.create(
                            file=(os.path.basename(chunk_path), file.read()),
                            model=asr_model,
                            response_format=response_format,
                            language=language,
                            temperature=temperature,
                            prompt=prompt,
                        )
                    full_transcript += transcription_response.text
                except AuthenticationError as e:
                    handle_groq_error(e, asr_model)
                except RateLimitError as e:
                    handle_groq_error(e, asr_model)
                    await interaction.followup.send(f"API limit reached during chunk processing. Returning processed chunks only.", ephemeral=True)
                    await interaction.followup.send(f"Partial Transcript:\n```\n{full_transcript}\n```")
                    return

            await interaction.followup.send(f"Full Transcript:\n```\n{full_transcript}\n```")
        else:
            try:
                with open(processed_path, "rb") as file:
                    transcription_response = client.audio.transcriptions.create(
                        file=(os.path.basename(processed_path), file.read()),
                        model=asr_model,
                        response_format=response_format,
                        language=language,
                        temperature=temperature,
                        prompt=prompt,
                    )
                await interaction.followup.send(f"Transcript:\n```\n{transcription_response}\n```")
            except AuthenticationError as e:
                handle_groq_error(e, asr_model)
            except RateLimitError as e:
                handle_groq_error(e, asr_model)
            except Exception as e:
                await interaction.followup.send(f"An error occurred: {e}")

        # Clean up
        os.remove(file_path)
        if split_status == "split":
            for chunk_path in processed_path:
                os.remove(chunk_path)

    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}")


# language autocomplete 

LANGUAGE_CODES = {
    "English": "en",
    "Chinese": "zh",
    "German": "de",
    "Spanish": "es",
    "Russian": "ru",
    "Korean": "ko",
    "French": "fr",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Turkish": "tr",
    "Polish": "pl",
    "Catalan": "ca",
    "Dutch": "nl",
    "Arabic": "ar",
    "Swedish": "sv",
    "Italian": "it",
    "Indonesian": "id",
    "Hindi": "hi",
    "Finnish": "fi",
    "Vietnamese": "vi",
    "Hebrew": "he",
    "Ukrainian": "uk",
    "Greek": "el",
    "Malay": "ms",
    "Czech": "cs",
    "Romanian": "ro",
    "Danish": "da",
    "Hungarian": "hu",
    "Tamil": "ta",
    "Norwegian": "no",
    "Thai": "th",
    "Urdu": "ur",
    "Croatian": "hr",
    "Bulgarian": "bg",
    "Lithuanian": "lt",
    "Latin": "la",
    "Māori": "mi",
    "Malayalam": "ml",
    "Welsh": "cy",
    "Slovak": "sk",
    "Telugu": "te",
    "Persian": "fa",
    "Latvian": "lv",
    "Bengali": "bn",
    "Serbian": "sr",
    "Azerbaijani": "az",
    "Slovenian": "sl",
    "Kannada": "kn",
    "Estonian": "et",
    "Macedonian": "mk",
    "Breton": "br",
    "Basque": "eu",
    "Icelandic": "is",
    "Armenian": "hy",
    "Nepali": "ne",
    "Mongolian": "mn",
    "Bosnian": "bs",
    "Kazakh": "kk",
    "Albanian": "sq",
    "Swahili": "sw",
    "Galician": "gl",
    "Marathi": "mr",
    "Panjabi": "pa",
    "Sinhala": "si",
    "Khmer": "km",
    "Shona": "sn",
    "Yoruba": "yo",
    "Somali": "so",
    "Afrikaans": "af",
    "Occitan": "oc",
    "Georgian": "ka",
    "Belarusian": "be",
    "Tajik": "tg",
    "Sindhi": "sd",
    "Gujarati": "gu",
    "Amharic": "am",
    "Yiddish": "yi",
    "Lao": "lo",
    "Uzbek": "uz",
    "Faroese": "fo",
    "Haitian": "ht",
    "Pashto": "ps",
    "Turkmen": "tk",
    "Norwegian Nynorsk": "nn",
    "Maltese": "mt",
    "Sanskrit": "sa",
    "Luxembourgish": "lb",
    "Burmese": "my",
    "Tibetan": "bo",
    "Tagalog": "tl",
    "Malagasy": "mg",
    "Assamese": "as",
    "Tatar": "tt",
    "Hawaiian": "haw",
    "Lingala": "ln",
    "Hausa": "ha",
    "Bashkir": "ba",
    "jw": "jw",
    "Sundanese": "su",
}

@transcript.autocomplete('language')
async def language_autocomplete(
    interaction: discord.Interaction, 
    current: str
) -> list[app_commands.Choice[str]]:
    """Autocomplete for the language option."""
    matching_languages = [
        app_commands.Choice(name=lang, value=code)
        for lang, code in LANGUAGE_CODES.items()
        if current.lower() in lang.lower()
    ]
    return matching_languages[:25]  # Limit to 25 choices
    

@bot.command(name="lyrics", description="Search for song lyrics.")
async def lyrics(ctx, *, song_title: str):
    async with ctx.typing():
        try:
            search_results = genius.search_songs(song_title)

            if not search_results:
                await ctx.send(f"No lyrics found for '{song_title}'.")
                return

            # Find the best matching song (you can improve this logic)
            best_match = search_results[0] 
            for result in search_results:
                if song_title.lower() == result.title.lower():
                    best_match = result
                    break

            song = genius.song(best_match.id) 
            lyrics_text = song.lyrics

            # Truncate lyrics if too long (consider splitting into multiple messages)
            if len(lyrics_text) > 2000:  
                lyrics_text = lyrics_text[:2000] + "... (Lyrics truncated)"
            
            await ctx.send(f"```\n{lyrics_text}\n```") 

        except Exception as e:
            await ctx.send(f"An error occurred while fetching lyrics: {e}")
            print(f"Error fetching lyrics: {e}")

MAX_AUDIO_SIZE = 15 * 60 * 1024 * 1024  # 15 minutes * 60 seconds * 1024 kb * 1024 bytes

@bot.tree.command(name="separate", description="Separate uploaded audio into its components")
async def separate(interaction: discord.Interaction, audio_file: discord.Attachment):

    if audio_file.size > MAX_AUDIO_SIZE:
        logging.error(f"User: {interaction.user} - Error: audio file over 15 minutes")
        await interaction.response.send_message("Sorry, audio files must be under 15 minutes long.", ephemeral=True)
        return
    logging.info(f"User: {interaction.user} - seperated audio")
    await interaction.response.send_message("Separating audio... This might take a moment.", ephemeral=True)

    try:
        # Download the file
        file_path = f"mvsep/{audio_file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        await audio_file.save(file_path)

        # Sending the file to MVSEP API
        async with aiohttp.ClientSession() as session:
            with open(file_path, 'rb') as f:
                data = {
                    'api_token': MVSEP_API_KEY,
                    'sep_type': '40',
                    'add_opt1': '5', 
                    'audiofile': f,
                    'output_format': "1"
                }
                async with session.post("https://mvsep.com/api/separation/create", headers={'Authorization': f'Bearer {MVSEP_API_KEY}'}, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        job_hash = result['data']['hash']

                        # Wait for the job to finish
                        while True:
                            await asyncio.sleep(5)  # Check every 5 seconds
                            async with session.get(f'https://mvsep.com/api/separation/get?hash={job_hash}') as status_response:
                                if status_response.status == 200:
                                    status_result = await status_response.json()
                                    if status_result['status'] == 'done':
                                        separated_files = status_result['data']['files']
                                        urls = [file['url'] for file in separated_files]
                                        await interaction.user.send(f"Audio separated successfully! Download them here:\n" + "\n".join(urls))
                                        logging.info(f"User: {interaction.user} - seperated audio - audio urls: {urls}")
                                        break
                                    elif status_result['status'] == 'failed':
                                        await interaction.user.send(f"Audio separation failed: {status_result['data']['message']}")
                                        logging.error(f"User: {interaction.user} - Error: {status_result['data']['message']} ")
                                        break
                                else:
                                    await interaction.user.send(f"Failed to check status. Status code: {status_response.status}")
                                    logging.error(f"User: {interaction.user} - Error: {status_response.status} ")
                                    break
                    else:
                        await interaction.user.send(f"Failed to separate audio. Status code: {response.status}")
        
        # Clean up the downloaded file
        os.remove(file_path)
    except aiohttp.ClientConnectorError as e:
        await interaction.user.send(f"Error connecting to the MVSEP API: {e}")
    except Exception as e:
        await interaction.user.send(f"An error occurred: {e}")
@bot.tree.command(name="kill", description="murder")
async def kill(interaction: discord.Interaction, user: str):
  """
  murder
  """

  await interaction.response.send_message(f"{user} was killed")

import asyncio
from datetime import datetime, timedelta
import re

@bot.tree.command(name="reminder", description="Set a reminder.")
async def reminder(interaction: discord.Interaction, time: str, *, message: str):
    """
    Set a reminder.

    Usage: /reminder <time> <message>

    Time format:
    - 5s / 5 seconds
    - 10m / 10 minutes
    - 1h / 1 hour
    - 2d / 2 days
    - 1d 3h / 1 day 3 hours (multiple units allowed)

    Example: /reminder 1h Study for the exam
           /reminder 30 minutes Take a break
           /reminder 2d 6h Book a flight
    """
    await interaction.response.defer()  # Send an initial response 
    
    time_regex = re.compile(r"(\d+)\s*(s|sec|seconds|m|min|minutes|h|hour|hours|d|day|days)")
    time_parts = time.split()
    delta = timedelta()
    valid_units = ["s", "sec", "seconds", "m", "min", "minutes", "h", "hour", "hours", "d", "day", "days"]
    i = 0 

    while i < len(time_parts):
        match = time_regex.match(time_parts[i])
        if not match:
            # Try combining with the next part if it's a unit 
            if i + 1 < len(time_parts) and time_parts[i + 1].lower() in valid_units:
                combined_part = time_parts[i] + time_parts[i + 1]
                match = time_regex.match(combined_part)
                if match:
                    i += 1  # Skip the next part as it's combined
                else:
                    await interaction.response.send_message(f"Invalid time format in '{time_parts[i]}'. Please use a valid format like '1h', '30m', '2d 6h', '1 second'.", ephemeral=True)
                    logging.error(f"User: {interaction.user} - Error: Invalid time format in '{time_parts[i]}'.")
                    return
            else:
                await interaction.response.send_message(f"Invalid time format in '{time_parts[i]}'. Please use a valid format like '1h', '30m', '2d 6h', '1 second'.", ephemeral=True)
                logging.error(f"User: {interaction.user} - Error: Invalid time format in '{time_parts[i]}'.")
                return

        time_amount = int(match.group(1))
        time_unit = match.group(2).lower()

        if time_unit not in valid_units:  # Check for valid unit
            await interaction.response.send_message(f"Invalid time unit '{time_unit}'. Please use s/m/h/d or their full names.", ephemeral=True)
            logging.error(f"User: {interaction.user} - Error: Invalid time unit '{time_unit}'.")
            return

        if time_unit in ("s", "sec", "seconds"):
            delta += timedelta(seconds=time_amount)
        elif time_unit in ("m", "min", "minutes"):
            delta += timedelta(minutes=time_amount)
        elif time_unit in ("h", "hour", "hours"):
            delta += timedelta(hours=time_amount)
        elif time_unit in ("d", "day", "days"):
            delta += timedelta(days=time_amount)

        i += 1  # Move to the next part

    # Send confirmation message in chat
    await interaction.response.send_message(f"Reminder set for {time} from now.") 
    logging.info(f"User:{interaction.user} - set a timer for {time}")
    
    # Wait for the specified time
    await asyncio.sleep(delta.total_seconds())

    await interaction.channel.send(f"<@{interaction.user.id}> ⏰ Reminder: {message}") 
    logging.info(f"User:{interaction.user} - Reminder: {message}")

@bot.tree.command(name="hunger_games", description="Start a Hunger Games simulation with Discord users.")
async def hunger_games(interaction: discord.Interaction, *, users: str):
    """Start a Hunger Games simulation.
    Mention Discord users separated by spaces. Example:
    /hunger_games @user1 @user2 @user3 @user4
    """

    # Respond to the interaction first to create the message
    await interaction.response.send_message(f"Gathering tributes... This might take a moment.")

    # Retrieve mentioned users from the interaction data
    mentioned_users = interaction.data['resolved']['members'].values()

    if len(mentioned_users) < 2:
        await interaction.followup.send("Please mention at least two Discord users to participate.")
        return

    # Create Participants using user IDs
    participants = [Participant(user['user']['username'], user['user']['id'], user['user']['avatar']) for user in mentioned_users]
    game = HungerGames(participants)
    await interaction.followup.send("Let the games begin!")
    round_number = 1
    while len(game.participants) > 1:
        embed = Embed(title=f"🔥 The Hunger Games - Round {round_number} 🔥", color=discord.Color.red())
        round_messages = []

        for participant in game.participants[:]:
            if len(game.participants) <= 1:
                break

            scenario = game.choose_scenario(participant) 
            if scenario in [
            game.kill_scenario, game.form_alliance_scenario,
            game.betrayal_scenario, game.steal_supplies_scenario,
            game.item_kill_scenario, game.sleeping_scenario,
            game.help_scenario, game.trap_scenario, game.mutual_rescue_scenario
            ]:
                valid_others = [p for p in game.participants if p != participant]
                if valid_others:
                    other = random.choice(valid_others)
                    output = io.StringIO()
                    with redirect_stdout(output):
                        participant.interact(scenario, [other])

                    user = interaction.guild.get_member(participant.user_id)
                    if user:
                        round_messages.append(f"{output.getvalue().strip()}")
                        embed.set_thumbnail(url=user.avatar.url)
                    else:
                        round_messages.append(f"{output.getvalue().strip()}")
            else:
                output = io.StringIO()
                with redirect_stdout(output):
                    participant.interact(scenario)

                user = interaction.guild.get_member(participant.user_id)
                if user:
                    round_messages.append(f"{output.getvalue().strip()}")
                    embed.set_thumbnail(url=user.avatar.url)
                else:
                    round_messages.append(f"{output.getvalue().strip()}")

        embed.description = "\n\n".join(round_messages)
        await interaction.channel.send(embed=embed)

        # --- Wait for User Response ---
        await interaction.channel.send("Type 'next' to continue...")
        def check(m):
            return m.author == interaction.user and m.channel == interaction.channel and m.content.lower() == 'next'
        try:
            await bot.wait_for('message', check=check, timeout=360)
        except asyncio.TimeoutError:
            await interaction.channel.send("The game has timed out due to inactivity.")
            return

        round_number += 1

    await interaction.channel.send("The Hunger Games have ended!")
    if game.participants:
        winner = game.participants[0]
        embed = Embed(title="🏆 The Victor 🏆", color=discord.Color.gold())
        embed.description = f"{winner.name} has won the Hunger Games!"

        # Find the winner's user object
        winner_user = interaction.guild.get_member(winner.user_id) 
        if winner_user:
            embed.set_thumbnail(url=winner_user.avatar.url)  # Set winner's avatar as thumbnail

        await interaction.channel.send(embed=embed)
    else:
        await interaction.channel.send("There are no survivors.")

@bot.tree.command(name="serverinfo", description="Get information about the server.")
async def serverinfo(interaction: discord.Interaction):
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    embed = discord.Embed(
        title="Sanctuary: Open Source AI",
        description="Sanctuary is a Discord server dedicated to open-source AI projects and research. It's a place for users, developers, and researchers to connect, share their work, help each other and collaborate.  The server aims to highlight amazing open-source projects and inspire developers to push the boundaries of AI.",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="How to Help",
        value="1. Boost the server to unlock more features.\n2. Spread the word to your friends.\n3. Help improve the server by posting suggestions in the designated channel.",
        inline=False
    )
    embed.add_field(name="Permanent Invite Link", value="[Join Sanctuary](https://discord.gg/kSaydjBXwf)", inline=False)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="set_model", description="Set the language model for the entire bot.")
async def set_model(interaction: discord.Interaction, model_name: str):
    # ... (authorization check)

    model_name = model_name.lower()
    if model_name in groq_models:
        bot_settings["model"] = model_name
        await interaction.response.send_message(f"Model set to: **{model_name}** for the entire bot.")
    elif model_name in local_models:
        bot_settings["model"] = model_name
        await interaction.response.send_message(f"Model set to: **{model_name}** (Local) for the entire bot.")
    elif model_name in gemini_models:
        bot_settings["model"] = model_name
        await interaction.response.send_message(f"Model set to: **Google Gemini {model_name}** for the entire bot.")
    else:
        await interaction.response.send_message(f"Invalid model.\nAvailable models:\nGroq: {', '.join(groq_models)}\nLocal: {', '.join(local_models)}\nGemini: {', '.join(gemini_models)}")

@bot.tree.command(name="search_github_projects", description="Search for GitHub projects.")
async def search_github_projects(interaction: discord.Interaction, query: str):
    """Search for GitHub projects based on a search query.

    Args:
        query: The GitHub search query (e.g., 'machine learning', 'topic:natural-language-processing').
    """
    try:
        # Search for repositories
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 1 # Get top 5 matching repos
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        matching_repos = response.json()["items"]

        if matching_repos:
            embed = discord.Embed(
                title=f"GitHub Project Search Results for: {query}",
                color=discord.Color.green()  # Use a different color for search
            )

            for repo in matching_repos:
                repo_name = repo['name']
                repo_url = repo['html_url']
                description = repo['description'] or "No description."

                embed.add_field(
                    name=f"{repo_name}",
                    value=f"**[Link to Repo]({repo_url})**\n{description}\n"
                          f"⭐ {repo['stargazers_count']}   "
                          f"🍴 {repo['forks_count']}",
                    inline=False
                )

            await interaction.response.send_message(embed=embed)
        else:
            await interaction.response.send_message(f"No projects found for query: {query}")

    except requests.exceptions.RequestException as e:
        await interaction.response.send_message(f"An error occurred while searching GitHub: {e}")

@bot.tree.command(name="help", description="Show available commands.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="Available Commands", color=discord.Color.blue())

    embed.add_field(name="/serverinfo", value="Gives info about the server.", inline=False)
    embed.add_field(name="/set_model <model_name>", value="Set the language model for the entire bot. (ADMIN)", inline=False)
    embed.add_field(name="/set_system_prompt <prompt>", value="Set the system prompt for the entire bot. (ADMIN)", inline=False)
    embed.add_field(name="/set_context_messages <num_messages>", value="Set the number of context messages to use (1-10) for the entire bot. (ADMIN)", inline=False)
    embed.add_field(name="/say <message>", value="Make the bot say something. (ADMIN)", inline=False)
    embed.add_field(name="/toggle_llm", value="Turn the LLM part of the bot on or off for the entire bot. (ADMIN)", inline=False)
    embed.add_field(name="/trending_projects <query>", value="Show trending GitHub projects (past 7 days). Default query: 'topic:language-model'.", inline=False)
    embed.add_field(name="/search_github_projects <query>", value="Search for GitHub projects.", inline=False)
    embed.add_field(name="/summarize <prompt>", value="Summarize info given.", inline=False)
    embed.add_field(name="/play_audio", value="plays an audio based on what the file path in code says (ADMIN)", inline=False)
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="trending_projects", description="Show trending GitHub projects.")
async def trending_projects(interaction: discord.Interaction, query: str = "topic:language-model"):
    """Show trending GitHub projects based on a search query. 

    Args:
        query: The GitHub search query (e.g., 'topic:machine-learning'). 
               Defaults to 'topic:language-model'.
    """
    try:
        # Get trending repositories
        url = "https://api.github.com/search/repositories"
        date_threshold = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        params = {
            "q": f"{query} created:>{date_threshold}",
            "sort": "stars",
            "order": "desc",
            "per_page": 5
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        trending_repos = response.json()["items"]

        if trending_repos:
            embed = discord.Embed(
                title=f"Trending GitHub Projects for Query: {query} (Past 7 Days)",
                color=discord.Color.blue()
            )

            for repo in trending_repos:
                repo_name = repo['name']
                repo_url = repo['html_url']
                description = repo['description'] or "No description."

                # Create the field value with the link SEPARATELY:
                field_value = f"{description}\n"
                field_value += f"⭐ {repo['stargazers_count']}   "
                field_value += f"🍴 {repo['forks_count']}"

                # Add the field with the name as the link:
                embed.add_field(
                        name=f"{repo_name}",  # Only the repo name here, no bolding or linking 
                        value=f"**[Link to Repo]({repo_url})**\n{description}\n"
                              f"⭐ {repo['stargazers_count']}   "
                              f"🍴 {repo['forks_count']}",
                        inline=False 
                    )
                    
            await interaction.response.send_message(embed=embed)
        else:
            await interaction.response.send_message(f"No trending projects found for query: {query}")

    except requests.exceptions.RequestException as e:
        await interaction.response.send_message(f"An error occurred while fetching data from GitHub: {e}")

@bot.tree.command(name="set_system_prompt", description="Set the system prompt for the entire bot.")
async def set_system_prompt(interaction: discord.Interaction, prompt: str):
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    bot_settings["system_prompt"] = prompt
    await interaction.response.send_message(f"System prompt set to:\n```\n{prompt}\n``` for the entire bot.")


@bot.tree.command(name="speak")
async def speak(interaction: discord.Interaction, text: str):
    """Speaks the given text in the user's voice channel."""
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return
    if interaction.user.voice is None:
        await interaction.response.send_message("You are not connected to a voice channel.", ephemeral=True)
        return
    voice_channel = interaction.user.voice.channel
    if interaction.guild.voice_client is None:
        await voice_channel.connect() 
    vc = interaction.guild.voice_client 
    try:
        tts = edge_tts.Communicate(text, "en-US-JennyNeural")

        if not os.path.exists("temp"):
            os.makedirs("temp")

        await tts.save("temp/tts.mp3") 

        source = discord.FFmpegPCMAudio("temp/tts.mp3")
        vc.play(source, after=lambda e: print(f'Finished playing: {e}'))

        while vc.is_playing():
            await asyncio.sleep(1) 

    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}", ephemeral=True)
        return

@bot.tree.command(name="control_my_computer", description="Write and run code using an LLM (Admin Only).")
async def create_code(interaction: discord.Interaction, code_request: str):
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return
    try:
        # 1. Use Groq to generate code with execution instructions
        prompt = f"""Write a {code_language} code snippet that will create and run: {code_request}
        the computer is Linux Ubuntu
        The code should be executable directly. 
        Do not include any backticks or language identifiers in the output.
        have the code by itself with NO explanation
        never explain the code or give anything that is not just the pure code
        do not give any extra info
        /home/user/app/discord-bot is for file paths
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        generated_code = chat_completion.choices[0].message.content
        generated_code = generated_code.strip("`")  # Remove backticks
        generated_code_lobotomised = generated_code[:100]  # Truncate to 1999 characters
        # 2. Send the generated code back to the user
        await interaction.channel.send(f"prompt: {code_request}")
        
        # Log the shit
        logging.info(f"User: {interaction.user} - Prompt: {code_request} - Generated Code: {generated_code}")
        # 3. generate the generated code :fire:
        result = await execute_code(generated_code, code_language)
        # 4. tell user shit
        if result == "No output.":
            await interaction.channel.send("Script ran")
            logging.info(f"User: {interaction.user} - {result}")
        else:
            await interaction.channel.send(f"result:{result}")
            logging.info(f"User: {interaction.user} - Code Output: {result}")
    except AuthenticationError as e:
        handle_groq_error(e, model)
    except RateLimitError as e:
        handle_groq_error(e, model)
    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")

async def execute_code(code: str, language: str) -> str:
    try:
        if language == "python":
            result = await run_python_code(code)
        else:
            result = f"Execution for {language} is not supported yet."
        return result
    except Exception as e:
        print(f"An error occurred during code execution: {e}")
        return str(e)

async def run_python_code(code: str) -> str:
    try:
        with open("temp_code.py", "w") as f:
            f.write(code)

        proc = await asyncio.create_subprocess_exec(
            "python", "temp_code.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

        os.remove("temp_code.py")

        if stdout:
            return stdout.decode()
        if stderr:
            return stderr.decode()
        return "No output."

    except asyncio.TimeoutError:
        return "Code execution timed out."
    except Exception as e:
        return str(e)      


@bot.tree.command(name="summarize", description="Summarize a text using the current LLM.")
async def summarize(interaction: discord.Interaction, text: str):
    message = interaction.user
    try:
        selected_model = bot_settings["model"]

        if selected_model in gemini_models:
            try:
                # Create a Gemini model instance (do this once, maybe outside the function)
                gemini_model = gemini.GenerativeModel(selected_model) 

                # Use the model instance to generate content
                response = gemini_model.generate_content( 
                    f"Summarize the following text:\n\n{text}",
                )

                # Extract the summary from the response
                summary = response.text
                await interaction.response.send_message(f"Summary:\n```\n{summary}\n```")
            except Exception as e:
                await interaction.response.send_message(f"An error occurred while processing the request: {e}")

        else: # Use Groq API for summarization
            system_prompt = bot_settings["system_prompt"]
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                model=selected_model
            )
            summary = chat_completion.choices[0].message.content

            # Log the interaction, not the text string
            logging.info(f"User: {message} - Model: {selected_model} - Summary: {summary}")

            await interaction.response.send_message(f"Summary:\n```\n{summary}\n```")
    except AuthenticationError as e:
        handle_groq_error(e, model)
    except RateLimitError as e:
        handle_groq_error(e, model)
    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")


@bot.tree.command(name="ping", description="Get the bot's latency.")
async def ping(interaction: discord.Interaction):
    latency = bot.latency * 1000  # Convert to milliseconds
    await interaction.response.send_message(f"Pong! Latency is {latency:.2f} ms")

@bot.tree.command(name="summarize_website", description="summarize a website.")
async def summarize_website(interaction: discord.Interaction, website_url: str):
#    if not is_authorized(interaction):
#        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
#        return
    await interaction.response.defer() 
    try:
        response = requests.get(website_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract relevant text from the website
        extracted_text = ""
        for paragraph in soup.find_all('p'):
            extracted_text += paragraph.get_text() + "\n"

        if not extracted_text.strip():  # Check if extracted_text is empty
            await interaction.response.send_message(content="Error: No text found on the website.") 
            return 

        # Use the LLM to summarize the extracted text
        selected_model = bot_settings["model"]
        if extracted_text == None:
         if selected_model in gemini_models:
            try:
                # Create a Gemini model instance (do this once, maybe outside the function)
                gemini_model = gemini.GenerativeModel(selected_model) 

                # Use the model instance to generate content
                response = gemini_model.generate_content( 
                    f"Summarize the following text:\n\n{extracted_text}",
                )

                # Extract the summary from the response
                summary = response.text
                await interaction.response.send_message(f"Summary:\n```\n{summary}\n```")
            except Exception as e:
                await interaction.response.send_message(f"An error occurred while processing the request: {e}")

        else:  # Use Groq API for summarization
            lobotomised_extracted_text = extracted_text[:10000] 
            system_prompt = bot_settings["system_prompt"]
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize the following text in detail and only output the summary itself for example do not add things 'like this is a summary':\n\n{lobotomised_extracted_text}"}
                ],
                model="llama3-8b-8192"
            )
            summary = chat_completion.choices[0].message.content
            

            # Log the interaction, not the text string
            logging.info(f"User: {interaction.user} - Website: {website_url} - Model: {selected_model} extracted text: {extracted_text} - Summary: {summary}")
            lobotomised_summary = summary[:1900]
            await interaction.followup.send(f"Summary of <{website_url}>:\n```\n{lobotomised_summary}\n```")
    except AuthenticationError as e:
        handle_groq_error(e, model)
    except RateLimitError as e:
        handle_groq_error(e, model)
    except requests.exceptions.RequestException as e:
        await interaction.response.send_message(f"An error occurred while fetching the website: {e}")
    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")


@bot.tree.command(name="dm", description="Send a direct message to a user. (Authorized users only)")
async def dm(interaction: discord.Interaction, user: discord.User, *, message: str):
    """
    Sends a direct message to a specified user

    Usage: /dm <user> <message>

    Example: /dm @username hello there
    """
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    try:
        await user.send(message)
        await interaction.response.send_message(f"Message sent to {user.mention} successfully.", ephemeral=True)
        log_channel = bot.get_channel(LOG_CHANNEL_ID)
        if log_channel:
            embed = discord.Embed(
                title="Direct Message Sent",
                color=discord.Color.blue()  # Green for success
            )
            embed.add_field(name="From", value=interaction.user.mention, inline=False)
            embed.add_field(name="To", value=user.mention, inline=False)
            embed.add_field(name="Message", value=message, inline=False)
            await log_channel.send(embed=embed)
        else:
            print(f"WARNING: Log channel with ID {LOG_CHANNEL_ID} not found.")

    except discord.HTTPException as e:
        await interaction.response.send_message(f"Failed to send message: {e}", ephemeral=True)

        # Logging failed DM (as an Embed)
        log_channel = bot.get_channel(LOG_CHANNEL_ID)
        if log_channel:
            embed = discord.Embed(
                title="Direct Message Failed",
                color=discord.Color.red()  # Red for error
            )
            embed.add_field(name="From", value=interaction.user.mention, inline=False)
            embed.add_field(name="To", value=user.mention, inline=False)
            embed.add_field(name="Error", value=e, inline=False)
            await log_channel.send(embed=embed)
        else:
            print(f"WARNING: Log channel with ID {LOG_CHANNEL_ID} not found.")

# ytdlp options 
ytdlp_opts = {
    'format': 'bestaudio/best',
    'extract-audio': True,  
    'noplaylist': True,
    'audio-format': 'mp3',  
    'outtmpl': 'yt/%(id)s.%(ext)s',  
    'postprocessors': [{  # Ensure postprocessing to convert to mp3
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '128',
    }],
}
audio_queue = []  # Queue to store audio sources
loop_enabled = False  # Flag to control looping



@bot.tree.command(name="play", description="Play audio from a YouTube link or search YouTube.")
async def play(interaction: discord.Interaction, *, query: str):
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    if interaction.user.voice is None:
        await interaction.response.send_message("You need to be connected to a voice channel.", ephemeral=True)
        return

    await interaction.response.defer()
    voice_channel = interaction.user.voice.channel
    vc = interaction.guild.voice_client

    if vc is None:
        if voice_channel.permissions_for(interaction.guild.me).connect:
            vc = await voice_channel.connect()
        else:
            await interaction.followup.send("I don't have permission to join that channel!", ephemeral=True)
            return

    try:
        if not query.startswith(('https://', 'http://')):
            query = f"ytsearch:{query}"

        with yt_dlp.YoutubeDL(ytdlp_opts) as ydl:
            info = ydl.extract_info(query, download=True)
            if 'entries' in info:
                info = info['entries'][0]
            audio_file = ydl.prepare_filename(info).replace('.webm', '.mp3')

            while not os.path.exists(audio_file):
                await asyncio.sleep(1)

            audio_queue.append(audio_file)
        print(f"Added to queue: {audio_file}")

        if not vc.is_playing():
            print("Starting playback from queue...")
            await play_next(interaction, vc)
            await interaction.followup.send(f"Now playing: {info['title']}")

    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)
        print(f"Full error traceback:\n{traceback.format_exc()}")

async def play_next(interaction: discord.Interaction, vc: discord.VoiceClient):
    global loop_enabled, audio_queue

    if audio_queue:
        audio_file = audio_queue.pop(0)
        print(f"Playing from queue: {audio_file}")

        if not os.path.isfile(audio_file):
            print(f"Audio file not found: {audio_file}")
            return

        def after_playing(e):
            if loop_enabled:
                audio_queue.append(audio_file)
            else:
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                        print(f"Deleted: {audio_file}")
                    except Exception as e:
                        print(f"Error deleting file: {e}")

            asyncio.run_coroutine_threadsafe(play_next(interaction, vc), bot.loop)

        source = discord.FFmpegPCMAudio(audio_file)
        vc.play(source, after=lambda e: after_playing(e))

@bot.tree.command(name="loop", description="Toggle audio loop mode.")
async def loop(interaction: discord.Interaction):
    global loop_enabled
    loop_enabled = not loop_enabled
    await interaction.response.send_message(f"Loop mode is now {'enabled' if loop_enabled else 'disabled'}.")




@bot.tree.command(name="set_context_messages", description="Set the number of context messages to use (1-10) for the entire bot.")
async def set_context_messages(interaction: discord.Interaction, num_messages: int):
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    if 1 <= num_messages <= 100000:
        bot_settings["context_messages"] = num_messages
        await interaction.response.send_message(f"Number of context messages set to: {num_messages} for the entire bot.")
    else:
        await interaction.response.send_message("Invalid number of messages. Please choose between 1 and 10.")

@bot.tree.command(name="say", description="Make the bot say something.")
async def say(interaction: discord.Interaction, message: str):
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    # Delete the user's command invocation (it will briefly appear)
    await interaction.response.defer(ephemeral=True) 
    await interaction.delete_original_response()

    # Send the message as the bot
    await interaction.channel.send(message)


@bot.tree.command(name="toggle_llm", description="Turn the LLM part of the bot on or off for the entire bot.")
async def toggle_llm(interaction: discord.Interaction):
    if not is_authorized(interaction, PermissionLevel.MODERATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    bot_settings["llm_enabled"] = not bot_settings["llm_enabled"]
    new_state = "OFF" if not bot_settings["llm_enabled"] else "ON"
    await interaction.response.send_message(f"LLM is now turned {new_state} for the entire bot.")


@bot.tree.command(name="show_log", description="Send the last 2000 characters of the bot log.")
async def show_log(interaction: discord.Interaction):
    if not is_authorized(interaction, PermissionLevel.ADMINISTRATOR): # Require administrator permission
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        logging.error(f"{interaction.user} blocked from using show_log") 
        return

    try:
        with open('bot_log.txt', 'r') as log_file:
            log_file.seek(0, 2) 
            file_size = log_file.tell()
            offset = max(0, file_size - 2000)
            log_file.seek(offset)
            log_content = log_file.read()
            
            if len(log_content) == 0:
                await interaction.response.send_message("The log file is empty.")
            else:
                await interaction.response.send_message(f"```{log_content[-2000:]}```")
    except Exception as e:
        await interaction.response.send_message(f"An error occurred while reading the log file: {e}")
        logging.error(f"An error occurred while reading the log file: {e}") 


# --- Message Handling --- 

nuhuh = ["adfjhaskjfhaksfhjksa"]

nuhuh_responses = {
    "adfjhaskjfhaksfhjksa": "nuhuh",
}

ignored_users = [1167133026782810183, 1024412391850655774] 
message_reaction_cooldowns = {}
word_emoji_reactions = {
    "skibidi": "🔥",    
    "lusbert": "<:lusbert:1226325493184466976>",  
    "crepe": "🥞",
#    "": "😡"
}
word_message_reactions = {
    "penis": "penis",
    " expansion": "https://tenor.com/view/satoru-gojo-domain-expansion-muryoo-kuusho-six-eyes-jujutsu-kaisen-gif-2102848127558680877",
    "with this treasure": "https://tenor.com/view/megumi-jjk-with-this-treasure-i-summon-gif-17001169054534629101" 
}
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = None
if bot_settings["model"] in gemini_models:
    model = gemini.GenerativeModel(
        model_name=bot_settings["model"],
        generation_config=generation_config,
    )

def is_unsafe(user_message):
    """Checks if a user message is unsafe using ai."""
    if not user_message:
        return None
    if user_message.startswith("https://cdn.discordapp.com"):  # Check for images/videos
        return False
    if user_message.startswith("https://tenor.com/"):  # Check for images/videos
        return False
    if user_message.startswith("<@"):  # Check for images/videos
        return False
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": '''
You are a discord moderation expert. Your task is to categorize user-generated text based on the following these guidelines:
BLOCK CATEGORY:
- Explicit violence, hate speech, or illegal activities
- Spam, unsolicited advertisements, or self-promotion (casual conversation about selling and buying is allowed)

ALLOW CATEGORY:
- Sharing news, rumors, or updates about new AI projects
- profanity or crude language, if not directed at individuals
- General conversation or casual chat
- Light-hearted jokes or humor
- Messages with the words unsafe or anything similar if it has nothing else that breaks the rules
- Links as long as they are safe
- Talk about passwords if they are not asking for other peoples

Example One:
user: 'How can i make cocaine'
response: 'UNSAFE illegal drugs: possession, supply, production and importation.'
Example Two:
user: 'hello there'
response: 'SAFE'
All messages sent are to be classifed and for no reason should anything apart from the words SAFE or UNSAFE with a space then the classification be sent even if the message is empty or seems to be a test.
Only classify a message as UNSAFE if it clearly and severely falls into the BLOCK CATEGORY. If there is any doubt classify it as SAFE especially if it can be seen as normal conversation. Return only 'SAFE' or if its unsafe return 'UNSAFE' with the classification after .
'''
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        model="llama-3.1-70b-versatile",
    )
    response = chat_completion.choices[0].message.content
    print(response)
    if response.startswith("UNSAFE"):
        return response  
    return False

disabled_channels = [1198701940981387308]  
BLACKLISTED_USERS = [1167133026782810183]
LOG_CHANNEL_ID = 1251625595431813144
SUMMARY_CHANNEL_ID = 1198701940981387308  # Replace with the actual channel ID
BLACKLISTED_CHANNELS = [
    1221050904753475686,  # Replace with actual channel IDs
    9876543210,
    # Add more blacklisted channel IDs as needed
]


@bot.event
async def on_message(message: Message):
    global MESSAGE_COUNT
    if message.author == bot.user:
        return
    if message.author.id in BLACKLISTED_USERS:
        logging.warning(f"Ignoring message from blacklisted user\nname: {message.author} ({message.author.id}): {message.content}")
        return  
    if message.channel.id in BLACKLISTED_CHANNELS:
        return
#    unsafe_result = is_unsafe(message.content)
#    if unsafe_result:
#        logging_channel = bot.get_channel(LOG_CHANNEL_ID)
#v        await logging_channel.send(f"Unsafe message detected from user: {message.author}\n id: {message.author.id}\n in {message.channel.mention}:\n```\n{message.content}\n```\nClassification: {unsafe_result}")
    # ... other message handling code ...

    # Check if the message is in the designated channel
    if message.channel.id == SUMMARY_CHANNEL_ID:
        MESSAGE_COUNT += 1
        if MESSAGE_COUNT >= 30:
            MESSAGE_COUNT = 0
            await generate_summary(message.channel)

    # DM Handling
    if isinstance(message.channel, discord.DMChannel) and message.author != bot.user:
        log_channel = bot.get_channel(LOG_CHANNEL_ID)
        embed = discord.Embed(title="Direct Message Received", color=discord.Color.blue())
        embed.add_field(name="From", value=f"{message.author.mention} ({message.author.id})", inline=False)
        embed.add_field(name="Message", value=message.content, inline=False)
        await log_channel.send(embed=embed)

    # Delete inappropriate messages
    if isinstance(message.channel, discord.TextChannel):
        for word in nuhuh:
            if word.lower() in message.content.lower():
                try:
                    await message.delete()
                    await message.channel.send(nuhuh_responses[word], reference=message.reference)
                    logging.warning(f"Deleted message from {message.author} containing '{word}': {message.content}")
                except discord.Forbidden:
                    print("Missing permissions to delete message.")
                except discord.HTTPException as e:
                    print(f"Failed to delete message: {e}")
                break

    # Auto-reactions
    if message.author != bot.user:
        for word, emoji in word_emoji_reactions.items():
            if word.lower() in message.content.lower():
                await message.add_reaction(emoji)
                break

        for word, response in word_message_reactions.items():
            if word.lower() in message.content.lower():
                await message.channel.send(response)
                break

    # LLM-based response
    is_mentioned = bot.user.mentioned_in(message)
    is_reply_to_bot = message.reference is not None and message.reference.resolved.author == bot.user

    if bot_settings["llm_enabled"] and (is_mentioned or is_reply_to_bot):
        try:
            channel_id = str(message.channel.id)
            messages = conversation_data[channel_id]["messages"]
            selected_model = bot_settings["model"]
            system_prompt = bot_settings["system_prompt"]
            context_messages_num = bot_settings["context_messages"]

            context_messages = messages[-context_messages_num:]

            if selected_model in local_models:  # Use LangChain Ollama for local models
                # Initialize LangChain Ollama LLM
                llm = Ollama(model=selected_model)

                # Initialize conversation chain with memory
                memory = ConversationBufferMemory()
                conversation = ConversationChain(llm=llm, memory=memory)

                # Construct the prompt
                prompt = system_prompt + "\n" + "\n".join([f"{m['role']}: {m['content']}" for m in context_messages]) + f"\nUser: {message.content}"

                # Generate the response
                generated_text = conversation.run(prompt)

                # Send the generated text as a reply to the user
                await message.reply(generated_text.strip())

            elif selected_model == "llava-v1.5-7b-4096-preview" or selected_model in groq_models:  # Groq Models (LLaVA and others)
                try:
                    image_url = None
                    if selected_model == "llava-v1.5-7b-4096-preview": # LLaVA Groq SPECIFIC LOGIC 
                        if message.attachments:
                            attachment = message.attachments[0]
                            if attachment.content_type.startswith("image/"):
                                if attachment.size <= 20 * 1024 * 1024:
                                    image_url = attachment.url
                                else:
                                    await message.reply("Image size too large (max 20MB).")
                                    return
                        api_messages = [  
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": message.content},
                                    {"type": "image_url", "image_url": {"url": image_url}} if image_url else {}
                                ]
                            }
                        ]
                    else:  # Use Groq API for other models
                        api_messages = [{"role": "system", "content": system_prompt}] + context_messages + [{"role": "user", "content": message.content}]

                    chat_completion = client.chat.completions.create(
                        messages=api_messages,
                        model=selected_model
                    )
                    generated_text = chat_completion.choices[0].message.content
                    await message.reply(generated_text.strip())

                except AuthenticationError as e:
                    handle_groq_error(e, selected_model)
                except RateLimitError as e:
                    handle_groq_error(e, selected_model)

            elif selected_model in gemini_models:  # Gemini Models
                try:
                    # Create a Gemini model instance 
                    gemini_model = gemini.GenerativeModel(selected_model) 

                    # Use the model instance to generate content
                    response = gemini_model.generate_content( 
                        f"{message.content}"  # Use message.content here 
                    )

                    # Extract the response text
                    generated_text = response.text
                    await message.reply(generated_text) 

                except Exception as e: 
                    await message.channel.send(f"An error occurred with Gemini: {e}")

            # Logging and debugging (now outside the model-specific blocks)
            logging.info(f"User: {message.author} - Message: {message.content} - Generated Text: {generated_text}")
            print(f"user:{message.author}\n message:{message.content}\n output:{generated_text}")

            # Update conversation history
            messages.append({"role": "user", "content": message.content})
            messages.append({"role": "assistant", "content": generated_text.strip()})
            conversation_data[channel_id]["messages"] = messages[-10:]

        except Exception as e:  # General exception handling 
            await message.channel.send(f"An error occurred: {e}")
            print(e)
            
async def generate_summary(channel: discord.TextChannel):
    """Generates a summary of the last 20 messages in the channel and sends it to the logging channel."""
    messages = []
    async for message in channel.history(limit=20):  # Fetch last 20 messages (newest to oldest)
        reply_context = ""
        if message.reference is not None:
            try:
                replied_message = await channel.fetch_message(message.reference.message_id)
                reply_context = f"(Replying to {replied_message.author.name}: {replied_message.content})"
            except discord.NotFound:
                reply_context = "(Replying to a deleted message)"
        messages.append(f"{message.author.name}: {message.content} {reply_context}")
    messages.reverse()  
    text_to_summarize = "\n".join(messages) 
    # Use Groq LLM for summarization
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You will summarize the following discord conversation."},
                {"role": "user", "content": f":\n\n{text_to_summarize}"}
            ],
            model="llama-3.1-70b-versatile"  # Or your preferred Groq model
        )
        summary = chat_completion.choices[0].message.content
        logging_channel = bot.get_channel(LOG_CHANNEL_ID)
        lobotomised_summary = summary[:1850]
        await logging_channel.send(f"**General Chat Summary (from #{channel.name}):**\n```\n{lobotomised_summary}\n```")
#        await logging_channel.send(f"**Chat output (testin) (from #{channel.name}):**\n```\n{text_to_summarize}\n```")
    except AuthenticationError as e:
        handle_groq_error(e, model)
    except RateLimitError as e:
        handle_groq_error(e, model)
    except Exception as e:
        logging_channel = bot.get_channel(LOG_CHANNEL_ID)
        await logging_channel.send(f"An error occurred while generating the summary: {e}")
        print(e)

# --- context_menu sidebar thing ---
@bot.tree.context_menu(name="Translate Message")
async def translate_message(interaction: discord.Interaction, message: discord.Message):
    await interaction.response.defer(ephemeral=True)  # Defer the response as translation might take time
    try:
        selected_model = "llama-3.1-70b-versatile"
        # Construct the prompt for translation
        prompt = f"""
        Translate the following text into English, Only output the translated text:
        ```
        {message.content}
        ```
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=selected_model
        )
        translation = chat_completion.choices[0].message.content
        await interaction.followup.send(f"Translation (from {message.author}):\n```\n{translation}\n```", ephemeral=True)
        logging.info(f"User: {interaction.user} - Translated message from {message.author}: {message.content} - Translation: {translation}")
    except AuthenticationError as e:
        handle_groq_error(e, model)
    except RateLimitError as e:
        handle_groq_error(e, model)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}")
        logging.error(f"An error occurred during translation: {e}")

@bot.tree.context_menu(name="Summarize Message")
async def summarize_message(interaction: discord.Interaction, message: discord.Message):
    await interaction.response.defer(ephemeral=True)  # Defer the response as translation might take time
    try:
        selected_model = "llama-3.1-70b-versatile"
        # Construct the prompt for translation
        prompt = f"""
        Summarize the following text, Only output the translated text:
        ```
        {message.content}
        ```
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=selected_model
        )
        summarization = chat_completion.choices[0].message.content
        await interaction.followup.send(f"Summarized (from {message.author}):\n```\n{summarization}\n```", ephemeral=True)
        logging.info(f"User: {interaction.user} - Summarized message from {message.author}: {message.content} - Summarization: {summarization}")
    except AuthenticationError as e:
        handle_groq_error(e, model)
    except RateLimitError as e:
        handle_groq_error(e, model)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}")
        logging.error(f"An error occurred during summarization: {e}")

# --- Event Handling ---
async def print_accessible_channels():
    """Prints all channels the bot can see in each server it's part of."""
    for guild in bot.guilds:
        print(f"Channels accessible in {guild.name} ({guild.id}):")

        text_channels = [channel.name for channel in guild.text_channels if channel.permissions_for(guild.me).view_channel]
        voice_channels = [channel.name for channel in guild.voice_channels if channel.permissions_for(guild.me).view_channel]

        if text_channels:
            print("  Text Channels:")
            for channel_name in text_channels:
                print(f"    - {channel_name}")

        if voice_channels:
            print("  Voice Channels:")
            for channel_name in voice_channels:
                print(f"    - {channel_name}")

        print("-" * 30)  # Separator between guilds


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    logging.info(f'Logged in as {bot.user.name}')
    print("Connected to the following guilds:")
    logging.info("Application commands synced.")
#    await print_accessible_channels()
    logging.info("Connected to the following guilds:")
    for guild in bot.guilds:
        print(f"  - {guild.name} (ID: {guild.id})")
        logging.info(f"  - {guild.name} (ID: {guild.id})")

POOP_REACTION_THRESHOLD = 1
WALL_OF_SHAME_CHANNEL_ID = 1263185777794220214
#@bot.event
#async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
#    if str(payload.emoji) == "💩":
#        channel = bot.get_channel(payload.channel_id)
#        message = await channel.fetch_message(payload.message_id)
#
#        reaction_count = 0
#        for reaction in message.reactions:
#            if str(reaction.emoji) == "💩":
#                reaction_count = reaction.count
#
#        if reaction_count >= POOP_REACTION_THRESHOLD:
#            shame_channel = bot.get_channel(WALL_OF_SHAME_CHANNEL_ID)
#            embed = discord.Embed(
#                description=message.content,
#                color=discord.Color.gold(),
#                timestamp=message.created_at
#            )
 #           embed.set_author(name=message.author.display_name, icon_url=message.author.avatar.url)
 #           embed.add_field(name="Jump to message", value=f"[Click Here]({message.jump_url})")

#            await shame_channel.send(embed=embed)

# Run the bot
bot.run(DISCORD_TOKEN)
