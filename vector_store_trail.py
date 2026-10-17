from dotenv import load_dotenv
load_dotenv()
import tempfile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
SAMPLE_DOCS = [


    Document(
        page_content="""
The Amazon rainforest is home to an extraordinary diversity of snakes, ranging from small arboreal species to large terrestrial and aquatic predators. The combination of dense vegetation, warm temperatures, high humidity, rivers, wetlands, and abundant prey creates suitable habitats for many different snake species. Some snakes spend most of their lives in trees, while others remain close to the forest floor, streams, swamps, or riverbanks. Their diets can include frogs, lizards, birds, rodents, fish, and other small animals.

One of the most recognizable snakes associated with the Amazon is the green anaconda. It is a large, heavy-bodied snake that spends much of its time in slow-moving water, swamps, flooded forests, and marshes. Its body is adapted to an aquatic lifestyle, with eyes and nostrils positioned high on the head, allowing it to remain mostly submerged while still observing its surroundings. Green anacondas are powerful constrictors and can prey on fish, birds, turtles, mammals, and other animals that come close to the water.

Although the green anaconda is often portrayed as an aggressive predator that actively hunts humans, this reputation is exaggerated. Like many wild animals, anacondas generally prefer to avoid unnecessary confrontation. Their hunting strategy relies heavily on ambush, patience, and their ability to remain concealed in vegetation or water. When suitable prey approaches, the snake can strike rapidly and use its muscular body to constrict the animal before swallowing it.

The Amazon ecosystem demonstrates how snake species can occupy very different ecological niches. Aquatic species such as the green anaconda are strongly associated with wetlands and rivers, while arboreal snakes may spend much of their time among branches and foliage. These differences in habitat, body structure, feeding behavior, and hunting strategy allow multiple snake species to coexist within the same rainforest ecosystem.
""",
        metadata={"source": "amazon_snakes_overview", "topic": "rainforest_ecology"},
    ),

    Document(
        page_content="""
The green anaconda, scientifically known as Eunectes murinus, is one of the largest snakes found in South America and is particularly associated with the Amazon Basin. Unlike many snakes that are primarily terrestrial or arboreal, the green anaconda is highly adapted to life in water. It can be found in rivers, streams, flooded forests, marshes, swamps, and other areas where aquatic environments provide suitable cover and prey.

The species has a thick, muscular body covered with dark green coloration and darker rounded markings. This pattern provides camouflage in the vegetation and muddy water where the snake often hunts. Its eyes and nostrils are positioned toward the top of its head, an adaptation that allows the animal to remain largely submerged while keeping its sensory organs above the surface. This makes the snake particularly effective at waiting for prey without exposing most of its body.

Green anacondas are constrictors rather than venomous snakes. Instead of injecting venom, they capture prey using their teeth and then wrap their muscular bodies around it. The pressure created by constriction can prevent the prey from breathing effectively, allowing the snake to eventually consume it. Depending on its size and the available food, an anaconda may consume fish, birds, turtles, capybaras, caimans, and other animals.

Anacondas do not need to eat frequently after consuming a large meal. Their metabolism and feeding behavior allow them to spend significant periods digesting food before hunting again. Their ability to remain hidden and wait for suitable prey is an important part of their survival strategy in the Amazon's complex wetland environments.

Despite their enormous size and strength, green anacondas are still vulnerable to changes in their environment. Destruction of wetlands, loss of vegetation, pollution, and human disturbance can reduce the quality of their habitat. Protecting the rivers, wetlands, and flooded forests of the Amazon therefore benefits not only anacondas but also many other species that depend on these ecosystems.
""",
        metadata={"source": "green_anaconda_guide", "topic": "green_anaconda"},
    ),

    Document(
        page_content="""
The emerald tree boa is an arboreal snake found in parts of the Amazon rainforest and other forested regions of South America. Unlike the green anaconda, which is strongly associated with aquatic environments, the emerald tree boa spends much of its life above the ground. Its bright green coloration helps it blend into leaves and branches, making it difficult for both predators and prey to detect.

Emerald tree boas have a distinctive body structure suited for life in trees. Their tails are prehensile, meaning that the tail can grip branches and help stabilize the snake while it moves through the canopy. Their large eyes are useful for detecting movement in relatively low-light forest conditions. The snake can remain coiled around a branch for long periods while waiting for suitable prey to approach.

Their diet includes small mammals, birds, frogs, and lizards. An emerald tree boa may use an ambush strategy, remaining motionless until prey comes within striking distance. Once it strikes, the snake uses its teeth and muscular body to hold the prey. Unlike venomous snakes, it relies primarily on constriction to subdue its prey.

The emerald tree boa also demonstrates why coloration should not be interpreted simply as decoration. In a dense rainforest environment, green coloration can provide effective camouflage against leaves and vegetation. This adaptation can help the snake remain hidden while hunting and can also reduce the chance of being detected by predators.

Arboreal snakes such as the emerald tree boa occupy ecological niches that differ significantly from those of aquatic snakes. While anacondas depend heavily on wetlands and waterways, emerald tree boas depend on vegetation and the three-dimensional structure of the forest canopy. The loss of mature rainforest trees can therefore affect species like the emerald tree boa even when nearby water sources remain intact.
""",
        metadata={"source": "emerald_tree_boa_guide", "topic": "emerald_tree_boa"},
    ),

    Document(
        page_content="""
The bushmaster is one of the most notable venomous snakes associated with tropical forests in northern South America. Unlike constrictors such as anacondas and boas, bushmasters use venom to immobilize their prey. They belong to the pit viper family and possess specialized venom-delivery structures that allow them to inject venom during a bite.

Bushmasters are generally terrestrial and are often associated with forest environments where leaf litter, fallen vegetation, and dense ground cover provide opportunities for camouflage. Their coloration and patterns can make them difficult to notice against the forest floor. This is particularly important for an ambush predator because remaining undetected allows the snake to wait for prey rather than actively searching for it.

The snake's diet can include rodents and other small mammals, as well as other suitable prey. Like many ambush predators, it can remain relatively motionless for long periods. When prey comes within range, the snake can strike quickly and inject venom. After the prey has been immobilized, the snake can locate and consume it.

Bushmasters are generally shy animals and encounters with people are uncommon compared with encounters involving some more widespread snake species. Nevertheless, because they are venomous, accidental encounters can be dangerous. Habitat protection and public awareness are important in areas where people and wildlife share forest environments.

The presence of a venomous predator such as the bushmaster illustrates the diversity of hunting strategies found within the Amazon rainforest. Different snakes have evolved different methods of capturing prey: anacondas rely on constriction and aquatic ambush, tree boas rely on arboreal ambush and constriction, while bushmasters combine terrestrial camouflage with venom-based predation.
""",
        metadata={"source": "bushmaster_guide", "topic": "venomous_snakes"},
    ),

    Document(
        page_content="""
The common lancehead, also known as Bothrops atrox, is a highly adaptable pit viper found across large areas of tropical South America, including parts of the Amazon rainforest. It is primarily terrestrial and can live in a variety of habitats ranging from dense forest to disturbed environments near human settlements. This adaptability has contributed to its relatively broad distribution compared with some highly specialized rainforest snakes.

The common lancehead is a venomous predator that commonly feeds on small mammals, amphibians, birds, and other available prey. Its hunting behavior often involves remaining concealed and waiting for prey to approach. The snake's coloration can provide camouflage among leaves, soil, branches, and other material on the forest floor.

Like other pit vipers, the common lancehead possesses heat-sensitive pit organs that can help it detect warm-blooded prey. These sensory adaptations are particularly useful in environments where visibility may be limited by dense vegetation or darkness. The snake can combine these sensory abilities with its sense of smell, vision, and other environmental information when locating prey.

The common lancehead is important to the rainforest food web because it acts as both predator and prey. By feeding on smaller animals, it influences populations of rodents and other organisms. At the same time, larger predators may prey on snakes, meaning that the species participates in several levels of the ecosystem.

Human encounters with the common lancehead can occur when agricultural expansion, roads, settlements, or other development bring people into areas occupied by the snake. Understanding its behavior and habitat is therefore important for reducing unnecessary conflict between humans and wildlife.
""",
        metadata={"source": "common_lancehead_guide", "topic": "pit_vipers"},
    ),

    Document(
        page_content="""
The Amazon rainforest contains many snake species that differ in size, appearance, habitat, diet, and defensive behavior. Comparing these species provides an interesting example of ecological specialization. The green anaconda is strongly associated with water and uses its powerful body to constrict prey. The emerald tree boa spends much of its time in vegetation and uses its prehensile tail to navigate branches. Bushmasters and common lanceheads are venomous terrestrial snakes that rely on camouflage and venom to capture prey.

Snake coloration is another important feature of rainforest survival. Bright green coloration can provide camouflage in the canopy, while brown or patterned scales can help terrestrial snakes disappear among leaf litter and fallen vegetation. These adaptations are particularly valuable for predators that depend on ambush hunting. Remaining unnoticed can reduce the energy required to actively search for prey and can increase the likelihood of a successful strike.

The snakes of the Amazon are also connected to the broader health of the rainforest ecosystem. They participate in food webs by controlling populations of rodents, amphibians, birds, fish, and other animals. Their presence can therefore influence ecological relationships beyond the individual species. At the same time, snakes themselves depend on healthy forests, wetlands, prey populations, and suitable shelter.

Deforestation and habitat fragmentation can affect snakes in different ways depending on their ecological requirements. Species that depend on large trees may be affected by canopy loss, while aquatic species can be affected by changes to rivers, wetlands, and water quality. Terrestrial snakes may face increased encounters with humans when forests are converted into roads, farms, and settlements.

Studying different snake species therefore provides more than an opportunity to learn about individual animals. It also demonstrates concepts such as adaptation, ecological niches, predator-prey relationships, camouflage, biodiversity, and habitat dependence. These characteristics make rainforest snakes useful examples for understanding how organisms interact with complex ecosystems.
""",
        metadata={"source": "amazon_snake_comparison", "topic": "comparison_ecology"},
    ),

]

from langchain_ollama import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

def create_vector_store():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )
        print(
            f"Vector store created {vectorstore._collection.count()} documents and persisted."
        )
        query = "How does the anaconda's body help it hunt in water?"
        results = vectorstore.similarity_search_with_score(query, k=3, filter={"topic": "green_anaconda"})
        print(f"Similarity search results for query '{query}':")
        for i, (doc, score) in enumerate(results):
            final_score = 1 / (1 + score)  # Convert distance to similarity
            print(f"Result {i + 1}: {doc.page_content} (Similarity: {final_score:.4f}) , Source: {doc.metadata['source']}")
        query = "How does the green anaconda's body help it hunt in water?"

if __name__ == "__main__":
    create_vector_store()