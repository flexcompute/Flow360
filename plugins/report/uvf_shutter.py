    async def process_3d_images(
        self, cases: list[Case], fig_name: str, url: str, manifest: list, data_storage: str = '.'
    ) -> list[str]:
        @backoff.on_exception(backoff.expo, ValueError, max_time=300)
        async def _get_image(session: aiohttp.client.ClientSession, url: str, manifest: list[dict]):
            async with session.post(url, json=manifest) as response:
                if response.status == 503:
                    raise ValueError("503 response received.")
                else:
                    return await response.read()

        async with aiohttp.ClientSession() as session:
            tasks = []
            img_names = []
            existing_images = os.listdir()
            for case in cases:
                # Stops case_by_case re-requesting images from the server
                # Empty list doesn't cause issues with later gather or zip
                img_name = os.path.join(data_storage, fig_name + "_" + case.name + ".png")
                img_names.append(img_name)
                if img_name in existing_images:
                    continue

                task = _get_image(session=session, url=url, manifest=manifest)
                tasks.append(task)
            responses = await asyncio.gather(*tasks)

            for response, img_name in zip(responses, img_names):
                with open(img_name, "wb") as img_out:
                    img_out.write(response)

        return img_names



        # url = "https://localhost:3000/screenshot" Local url
        url = "https://uvf-shutter.dev-simulation.cloud/screenshot"

        # Load manifest from case - move to loop later
        # os.chdir("/home/matt/Documents/Flexcompute/flow360/uvf-shutter/server/src/manifests")
        with open(os.path.join(here, "b777.json"), "r") as in_file:
            manifest = json.load(in_file)

        # os.chdir(current_dir)

        img_list = asyncio.run(
            self.process_3d_images(cases, self.fig_name, url=url, manifest=manifest, data_storage=data_storage)
        )